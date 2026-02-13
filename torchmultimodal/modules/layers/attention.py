# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.utils.common import shift_dim


class SelfAttention(nn.Module):
    """Computes attention over the entire n-dimensional input.

    Args:
        attn_dropout (float, optional): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)`` where ``q_dn`` is the
                dimension of the flattened query input along its latent dimensions and ``k_dn`` that of the
                flattened key input. Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            Output tensor.
        """
        _, _, *shape, _ = q.shape

        # flatten to b, h, (d1, ..., dn), dim_q/dim_kv
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            attn_dropout=self.attn_dropout if self.training else 0.0,
        )

        return out.unflatten(2, shape)


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism and caching for fast decoding.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the model to jointly
    attend to information from different representation subspaces at different positions,
    as described in `"Attention Is All You Need (Vaswani et al. 2017)"<https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        dim_q (int): Dimensionality of query input. Also the embedding dimension of the model.
        dim_kv (int): Dimensionality of key/value input. Projects to the embedding dimension of the model, ``dim_q``.
        n_head (int): Number of attention heads.
        attn_module (nn.Module): Module of attention mechanism to use. Default is ``SelfAttention``.
            See :class:`~torchmultimodal.modules.layers.attention.SelfAttention` for API details.
        add_bias (bool): Whether to add bias to the q, k, v, linear layers or not. Default is ``True``.

    Attributes:
        cache (Dict[str, Tensor]): Dictionary that stores past key/value vectors.

    Raises:
        ValueError: When ``dim_q`` or ``dim_kv`` is not divisible by ``n_head``.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        n_head: int,
        attn_module: nn.Module = SelfAttention(),
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError(
                "The hidden size of q, k, v must be a multiple of the number of attention heads."
            )

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.n_head = n_head
        self.query = nn.Linear(dim_q, dim_q, bias=add_bias)  # q
        self.key = nn.Linear(dim_kv, dim_q, bias=add_bias)  # k
        self.value = nn.Linear(dim_kv, dim_q, bias=add_bias)  # v
        self.output = nn.Linear(dim_q, dim_q, bias=True)  # c

        self.attn = attn_module

        self.cache: Optional[Dict[str, Tensor]] = None

    def forward(
        self,
        q: Tensor,
        kv: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        **attn_kwargs: Any,
    ) -> Tensor:
        """
        Args:
            q (Tensor): Query of shape ``(b, d1, ..., dn, dim_q)`` or ``(b, seq_len, dim_q)``
                (for autoregressive decoding it's typical to pass in flattened tensors).
            kv (Tensor, optional): Key (and value) of shape ``(b, d1', ..., dn', dim_kv)`` or
                ``(b, seq_len', dim_kv)``. If this argument is specified, cross-attention will be applied.
                Default is ``None``.
            use_cache (bool): If ``True``, caches past ``k`` and ``v`` tensors for faster decoding.
                If ``False``, recomputes ``k`` and ``v`` for each decoding step. Default is ``False``.
            causal (bool): Whether to use causal attention or not. Default is ``False``.

        Returns:
            Output tensor.
        """
        # If kv is specified use those inputs for cross-attention, otherwise use q
        k = v = q if kv is None else kv
        # compute q
        q = split_multihead(self.query(q), self.n_head)

        # For causal k, v are provided step-wise so we should always compute them
        # For non-causal skip computing k, v if they have been cached
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)

        # fast decoding by caching past key, value tensors
        if use_cache:
            if not self.cache:
                # initialize the cache with the present k, v
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    # append present k, v to past k, v
                    # for autoregressive decoding inputs are flattened as 1D sequences
                    # so are the cached tensors: (b, n_heads, seq_len, c)
                    k_, v_ = self.cache["k"], self.cache["v"]
                    self.cache["k"] = torch.cat([k_, k], dim=2)
                    self.cache["v"] = torch.cat([v_, v], dim=2)
                # override the present k, v with the cache
                k, v = self.cache["k"], self.cache["v"]

        attn_out = self.attn(q, k, v, **attn_kwargs)
        a = merge_multihead(attn_out)
        a = self.output(a)

        return a


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_mask: Optional[Tensor] = None,
    attn_dropout: float = 0.0,
) -> Tensor:
    """Computes scaled dot-product attention using ``F.scaled_dot_product_attention``
    with the MATH backend, which follows the same computation path as the manual
    implementation (matmul -> scale -> mask -> softmax -> dropout -> matmul).

    The attention_mask uses a boolean-style convention where 1 means "attend" and
    0 means "mask". This is converted to the additive float mask format expected by
    ``F.scaled_dot_product_attention`` (0.0 for attend, -inf for mask).

    Args:
        q (Tensor): Query of shape ``(b, h, d1, ..., dn, dim_qk)`` or ``(b, h, seq_len, dim_qk)``.
        k (Tensor): Key of shape ``(b, h, d1', ...., dn', dim_qk)`` or ``(b, h, seq_len', dim_qk)``.
        v (Tensor): Value of shape ``(b, h, d1', ..., dn', dim_v)`` or ``(b, h, seq_len', dim_v)``.
        attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions.
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.

    Returns:
        Output tensor.
    """
    if attention_mask is not None:
        # Convert boolean-style mask (1=attend, 0=mask) to additive float mask
        # expected by F.scaled_dot_product_attention (0.0=attend, -inf=mask)
        attention_mask = torch.zeros_like(attention_mask, dtype=q.dtype).masked_fill(
            attention_mask == 0, float("-inf")
        )

    # F.scaled_dot_product_attention expects 4D inputs (b, h, seq_len, dim).
    # For n-dimensional inputs (b, h, d1, ..., dn, dim), fold the extra spatial
    # dims (d1, ..., dn-1) into the batch dimension, keeping only the last
    # spatial dim as the sequence dim.
    needs_reshape = q.dim() > 4
    if needs_reshape:
        q_shape = q.shape
        # Fold (b, h, d1, ..., dn-1) into one batch dim, keep (dn, dim)
        batch_dims = q_shape[:-2]
        q = q.reshape(-1, *q_shape[-2:])[:, None]  # (B', 1, dn, dim)
        k = k.reshape(-1, *k.shape[-2:])[:, None]
        v = v.reshape(-1, *v.shape[-2:])[:, None]
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(-1, *attention_mask.shape[-2:])[
                :, None
            ]

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=attn_dropout
        )

    if needs_reshape:
        # Restore shape: (B', 1, dn, dim_v) -> (b, h, d1, ..., dn, dim_v)
        out = out.squeeze(1).reshape(*batch_dims, *out.shape[-2:])

    return out


def split_multihead(x: Tensor, n_head: int) -> Tensor:
    """Splits channel dimension of input tensor of size (b, d1, ..., dn, c)
    into multiple heads, (b, n_head, d1, ..., dn, c // n_head)"""
    x = x.unflatten(-1, (n_head, -1))
    # Rearrange to put head dim first, (b, n_head, d1, ..., dn, c // n_head)
    x = shift_dim(x, -2, 1)
    return x


def merge_multihead(x: Tensor) -> Tensor:
    """Moves head dim back to original location and concatenates heads
    (b, n_head, d1, ..., dn, c // n_head) -> (b, d1, ..., dn, c)"""
    return shift_dim(x, 1, -2).flatten(start_dim=-2)
