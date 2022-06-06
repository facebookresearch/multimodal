# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.utils.common import shift_dim


class MultiHeadAttention(nn.Module):
    """Compute multihead attention with flexible attention function.

    Multihead attention linearly projects queries, keys, and values into an embedding
    space, which is divided into multiple 'heads'. Attention is computed for each head
    individually instead of across the entire embedding space once. This enables more
    varied representations and allows the model to jointly attend to information from
    different representation subspaces at different positions, as described in
    Attention Is All You Need (Vaswani et al. 2017).

    Args:
        shape (Tuple): shape of input data (d1 x ... x dn)
        dim_q (int): dimensionality of input into query weights
        dim_kv (int): dimensionality of input into key and value weights
        n_head (int): number of attention heads
        n_layer (int): ?
        causal (bool): use causal attention or not
        attn_module (nn.Module): module of attention function to use

    Inputs:
        q, k, v (Tensor): a [b, d1, ..., dn, c] tensor or
                          a [b, 1, ..., 1, c] tensor if decode_step is not None

    """

    def __init__(
        self,
        shape: Tuple,
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
        # Splits input tensor of size (b x (d1...dn) x hidden)
        # into (b x (d1...dn) x n_head x emb_dim)
        x = x.unflatten(-1, (self.n_head, -1))
        # Rearrange to put head dim first, (b x n_head x (d1...dn) x emb_dim)
        x = shift_dim(x, -2, 1)
        return x

    def _combine_multihead(self, x: Tensor) -> Tensor:
        # Moves head dim back to original location and concatenates heads
        # (b x n_head x (d1...dn) x emb_dim) -> (b x (d1...dn) x hidden)
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
                    )  # B x n_head x 3 x 32 x 32 x d
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
        shape (Tuple): shape of input data (d1 x ... x dn)
        causal (bool): use causal attention or not
        attn_dropout (float): probability of dropout after softmax

    Inputs:
        q, k, v (Tensor): a [b, d1, ..., dn, c] tensor or
                          a [b, 1, ..., 1, c] tensor if decode_step is not None

    """

    def __init__(
        self, shape: Tuple, causal: bool = False, attn_dropout: float = 0.0
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
        mask = torch.Tensor(self.mask) if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        elif mask is not None and q.size(2) < mask.size(0):
            mask = mask[range(q.size(2)), :][:, range(q.size(2))]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, attn_dropout=self.attn_dropout, training=self.training
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

    def __init__(self, axial_dim: int) -> None:
        super().__init__()
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

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
    training: bool = True,
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
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x c
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x c

    return a
