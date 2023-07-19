# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Optional, Tuple, Union

import torch

import torch.nn.functional as F
from torch import nn, Tensor


class MHAWithCacheOutput(NamedTuple):
    attn_output: Tensor
    past_key_value: Tuple[Tensor, Tensor]


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


class MultiHeadAttentionWithCache(nn.Module):
    """
    MultiHeadAttention module for both self-attention(SA) and cross-attention(CA).
    This class supports a cache mechanism for decoders to store previous states through
    "past_key_value". Key/value states should be only cached for self-attention cases.
    q, k, v share the same dimension for self-attention,
    but different for cross-attention, CA requires encoder hidden states dim as k, v dims.

    Args:
        dim_q (int): query embedding dimension
        dim_kv (int): key, value embedding dimension,
            same as dim_q for SA; equals to encoder dimension for cross-attention
        num_heads (int): number of attention heads
        dropout (float): dropout rate
        add_bias (bool): if true, adds a learnable bias to query, key, value.
            Defaults to True.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        dropout: float = 0.0,
        add_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim_q, dim_q, bias=add_bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.output_proj = nn.Linear(dim_q, dim_q)
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = False,
        use_cache: bool = False,
    ) -> Union[Tensor, MHAWithCacheOutput]:
        """
        Args:
            query (Tensor): input query of shape bsz x target_seq_len x embed_dim
            key (Tensor): key of shape bsz x source_seq_len x embed_dim
            value (Tensor): value of shape bsz x source_seq_len x embed_dim
            attn_mask (optional Tensor): Attention mask of shape bsz x target_seq_len x source_seq_len.
                Two types of masks are supported. A boolean mask where a value of True
                indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            past_key_value (optional tuple of tensors): cached key and value with the same shape of key, value inputs.
                The size of tuple should be 2, where the first entry is for cached key and second entry is for cached value.
            is_causal (bool): If true, does causal attention masking, attn_mask should be set to None if this is set to True
                 is_causal is a hint that the mask is a causal mask, providing incorrect hints can result in incorrect execution.
            use_cache (bool): whether to use cache for key and value tensors

        Returns:
            if use_cache is off, return attn_output tensor of shape bsz x seq_len x embed_dim;
            otherwise return namedtuple with attn_output, cached key and value.
        """
        bsz = query.size(0)
        embed_dim = query.size(-1)
        head_dim = embed_dim // self.num_heads
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # bsz x seq_len x embed_dim => bsz x num_heads x seq_len x head_dim
        query = query.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        if key.size(0) != bsz or value.size(0) != bsz:
            raise ValueError("key and value should have the same bsz as query.")
        key = key.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)

        # concat key value with cached values
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)

        # turn off causal attention inside scaled_dot_product_attention, we handle it separately with attn_mask.
        attn = F.scaled_dot_product_attention(
            query, key, value, attn_mask, self.dropout, is_causal
        )
        attn = attn.transpose(1, 2).reshape(bsz, -1, embed_dim)

        # add dense layer after attention
        attn_output = self.output_proj(attn)
        if use_cache:
            return MHAWithCacheOutput(attn_output, (key, value))
        return attn_output
