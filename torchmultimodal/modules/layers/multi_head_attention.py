# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor


class MultiHeadSelfAttention(nn.Module):
    """
    Multihead self attention.
    Similar to the self attention variant of MHA in attention.py but uses the scaled_dot_product_attention from PyTorch
    (which uses flash or memory efficient version for certain conditions).
    TODO: merge this into attention.py once other models are ready to use it.

    Args:
        embed_dim (int): embedding dimension of the input
        num_heads (int): number of attn heads
        dropout (float): dropout rate
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Args:
            query (Tensor): input query of shape bsz x seq_len x embed_dim
            attn_mask (optional Tensor): attention mask of shape bsz x seq_len x seq_len. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element should take part in attention.
            A float mask of the same type as query that is added to the attention score.
            is_causal (bool): If true, does causal attention masking. attn_mask should be set to None if this is set to True

        Returns:
            attention output Tensor of shape bsz x seq_len x embed_dim
        """

        bsz = query.size(0)
        embed_dim = query.size(-1)
        projected_query = self.input_proj(query)
        query, key, value = projected_query.chunk(3, dim=-1)

        head_dim = embed_dim // self.num_heads
        # bsz x seq len x embed_dim => bsz x num_heads x seq len x head_dim
        query = query.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask, self.dropout, is_causal
        )
        attn = attn.transpose(1, 2).reshape(bsz, -1, embed_dim)

        attn_out = self.output_proj(attn)
        return attn_out
