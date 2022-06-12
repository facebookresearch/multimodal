# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.utils.common import shift_dim, tensor_slice


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the model to jointly
    attend to information from different representation subspaces at different positions,
    as described in Attention Is All You Need (Vaswani et al. 2017).

    Args:
        shape (Tuple[int]): shape of input data (d1, ..., dn)
        dim_q (int): dimensionality of query
        dim_kv (int): dimensionality of key/value
        n_head (int): number of attention heads
        n_layer (int): number of attention layers being used in higher level stack
        causal (bool): use causal attention or not
        attn_module (nn.Module): module of attention mechanism to use

    Inputs:
        q, k, v (Tensor): a [b, d1, ..., dn, c] tensor or
                          a [b, 1, ..., 1, c] tensor if decode_step is not None

    """

    # TODO: remove dependency on n_layer, higher level detail should not be a parameter

    def __init__(
        self,
        shape: Tuple[int],
        dim_q: int,
        dim_kv: int,
        n_head: int,
        n_layer: int,
        causal: bool,
        attn_module: nn.Module,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head
        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False)  # q
        self.w_qs.weight.data.normal_(std=1.0 / torch.sqrt(torch.tensor(dim_q)))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False)  # k
        self.w_ks.weight.data.normal_(std=1.0 / torch.sqrt(torch.tensor(dim_kv)))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False)  # v
        self.w_vs.weight.data.normal_(std=1.0 / torch.sqrt(torch.tensor(dim_kv)))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True)  # c
        self.fc.weight.data.normal_(std=1.0 / torch.sqrt(torch.tensor(dim_q * n_layer)))

        self.attn = attn_module

        self.cache: Dict[str, Tensor] = dict()

    def _split_multihead(self, x: Tensor) -> Tensor:
        # Splits input tensor of size (b x (d1, ..., dn) x hidden)
        # into (b x (d1...dn) x n_head x emb_dim)
        x = x.unflatten(-1, (self.n_head, -1))
        # Rearrange to put head dim first, (b x n_head x (d1, ..., dn) x emb_dim)
        x = shift_dim(x, -2, 1)
        return x

    def _combine_multihead(self, x: Tensor) -> Tensor:
        # Moves head dim back to original location and concatenates heads
        # (b x n_head x (d1, ..., dn) x emb_dim) -> (b x (d1, ..., dn) x hidden)
        return shift_dim(x, 1, -2).flatten(start_dim=-2)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, decode_step=None, decode_idx=None
    ) -> Tensor:
        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self._split_multihead(self.w_qs(q))
        k = self._split_multihead(self.w_ks(k))
        v = self._split_multihead(self.w_vs(v))

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (
                        q.shape[0],
                        n_head,
                        *self.shape,
                        self.d_k,
                    )
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(
                        k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                        v=torch.zeros(v_shape, dtype=v.dtype, device=q.device),
                    )
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (
                    slice(None, None),
                    slice(None, None),
                    *[slice(i, i + 1) for i in decode_idx],
                )
                self.cache["k"][idx] = k
                self.cache["v"][idx] = v
            k, v = self.cache["k"], self.cache["v"]

        a = self.attn(q, k, v, decode_step, decode_idx)
        a = self._combine_multihead(a)
        a = self.fc(a)

        return a


class FullAttention(nn.Module):
    """Computes attention over the entire flattened input.

    Args:
        shape (Tuple[int]): shape of input data (d1, ..., dn)
        causal (bool): use causal attention or not
        attn_dropout (float): probability of dropout after softmax

    Inputs:
        q, k, v (Tensor): a [b, d1, ..., dn, c] tensor or
                          a [b, 1, ..., 1, c] tensor if decode_step is not None

    """

    def __init__(
        self, shape: Tuple[int], causal: bool = False, attn_dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        if self.causal:
            seq_len = int(torch.prod(torch.tensor(shape)).item())
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, decode_step=None, decode_idx=None
    ) -> Tensor:
        mask = Tensor(self.mask) if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        elif mask is not None and q.size(2) < mask.size(0):
            mask = mask[range(q.size(2)), :][:, range(q.size(2))]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, attn_dropout=self.attn_dropout if self.training else 0.0
        )

        return out.unflatten(2, old_shape)


class AxialAttention(nn.Module):
    """Computes attention over a single axis of the input. Other dims are flattened
    into the batch dimension.

    Args:
        axial_dim (int): dimension to compute attention on, index by input dimensions
                         (i.e., 0 for first input dimension, 1 for second)

    Inputs:
        q, k, v (Tensor): a [b, h, d1, ..., dn, c] tensor or
                          a [b, h, 1, ..., 1, c] tensor if decode_step is not None

    """

    def __init__(self, axial_dim: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout
        self.axial_dim = axial_dim + 2  # account for batch, head

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, decode_step=None, decode_idx=None
    ) -> Tensor:
        # Ensure axial dim is within right dimensions, should be between head dim and embedding dim
        if self.axial_dim >= len(q.shape) - 1:
            raise ValueError("axial dim does not match input shape")

        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(
            q, k, v, attn_dropout=self.attn_dropout if self.training else 0.0
        )
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
) -> Tensor:
    """Similar to PyTorch Core's _scaled_dot_product_attention but generalized
    to handle n-dimensional input tokens (images, video) and support multihead.
    Computes attention as described in Attention Is All You Need (Vaswani et al. 2017)

    Inputs:
        q, k, v (Tensor): a [b, h, d1, ..., dn, c] tensor
    """

    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x (d1, ..., dn) x c
    attn = F.dropout(attn, p=attn_dropout)

    a = torch.matmul(attn, v)  # b x n_head x (d1, ..., dn) x c

    return a


class BroadcastedPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        shape (Tuple[int]): shape of raw data before batching and embedding
        embedding_dim (int): the size of each embedding vector

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``

    Inputs:
        x (Optional[Tensor]): flattened input data, e.g., ``(batch, time * height * width, embedding_dim)``.
        decode_step (Optional[int]): position of the data that requires decoding.
    """

    def __init__(
        self,
        shape: Tuple[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if embedding_dim % len(shape) != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} modulo len(shape) {len(shape)} is not zero"
            )

        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embedding_dim = embedding_dim

        self.embedding = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(
                    torch.randn(shape[i], embedding_dim // n_dim) * 0.01
                )
                for i in range(n_dim)
            }
        )

    @property
    def seq_len(self) -> int:
        """Dimension of flattened data, e.g., time * height * width"""
        return torch.prod(torch.tensor(self.shape)).item()

    @property
    def decode_idxs(self):
        """Indices along the dims of data, e.g., ``(time, height, width)``."""
        return list(itertools.product(*[range(s) for s in self.shape]))

    def _broadcast(self, i: int) -> Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.shape``. The embedding dim is not touched.

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"].shape)
            torch.Size([2, 3])
            >>> pos_emb.embedding["d_0"] = nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
            >>> out = pos_emb._broadcast(i=0)
            >>> print(out)
            tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]])
            >>> print(out.shape)
            (1, 2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f"d_{i}"]
        # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
        emb = emb.view(
            1,
            *itertools.repeat(1, i),
            self.shape[i],
            *itertools.repeat(1, (self.n_dim - i - 1)),
            -1,
        )
        # (1, *self.shape, -1)
        emb = emb.expand(1, *self.shape, -1)

        return emb

    def _decode(
        self, decode_step: int, embeddings: Tensor, x_shape: Tuple[int]
    ) -> Tensor:
        """Returns the embedding vector immediately before the decoding location."""
        decode_idx = self.decode_idxs[decode_step - 1]
        embeddings = tensor_slice(
            embeddings,
            [0, *decode_idx, 0],
            [x_shape[0], *itertools.repeat(1, self.n_dim), x_shape[-1]],
        )

        return embeddings

    def forward(
        self, x: Optional[Tensor] = None, decode_step: Optional[int] = None
    ) -> Tensor:
        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)

        embeddings = torch.cat(
            embeddings, dim=-1
        )  # concatenated embeddings: (1, *(shape), embedding_dim)

        if decode_step is not None:
            embeddings = self._decode(decode_step, embeddings, x.shape)
            # decoded embedding: (1, *repeat(1, len(shape)), embedding_dim)

        return embeddings.flatten(start_dim=1, end_dim=-2)
