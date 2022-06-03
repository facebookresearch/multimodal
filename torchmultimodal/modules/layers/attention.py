# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.utils.common import shift_dim


class MultiHeadAttention(nn.Module):
    """Compute multihead attention with flexible attention function

    Args:
        shape (Tuple): ?
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
    ):
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

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.w_qs(q).unflatten(-1, (n_head, d_k))
        k = self.w_ks(k).unflatten(-1, (n_head, d_k))
        v = self.w_vs(v).unflatten(-1, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) ->  (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

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

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a)  # (b x seq_len x embd_dim)

        return a


class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = int(torch.prod(shape).item())
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
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
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2  # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0.0, training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float("-inf"))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)  # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v)  # b x n_head x d1 x ... x dn x d

    return a
