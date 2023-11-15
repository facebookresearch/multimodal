# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.multi_head_attention import (
    MultiHeadAttentionWithCache,
)


class AttentionPooler(nn.Module):
    """
    Attention pooling layer: pools inputs to sequence length n_queries by performing
        cross-attention with learned query embeddings. Originally proposed in
        https://arxiv.org/abs/1810.00825. This implementation is based on the one
        in open_clip repo: https://tinyurl.com/4yj492sc.
    Args:
        input_embed_dim (int): Embedding dimension of inputs.
        output_embed_dim (int): Embedding dimension of outputs.
        n_head (int): Number of attention heads.
        n_queries (int): Number of queries. Defaults to 256
        layer_norm_eps (Optional[float]): Epsilon for layer norms. Defaults to 1e-5
    """

    def __init__(
        self,
        input_embed_dim: int,
        output_embed_dim: int,
        n_head: int,
        n_queries: int = 256,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, output_embed_dim))
        self.attn = MultiHeadAttentionWithCache(
            dim_q=output_embed_dim, dim_kv=input_embed_dim, num_heads=n_head
        )
        self.ln_q = nn.LayerNorm(output_embed_dim, layer_norm_eps)
        self.ln_k = nn.LayerNorm(input_embed_dim, layer_norm_eps)
        self.ln_post = nn.LayerNorm(output_embed_dim, layer_norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_embed_dim).
        Returns:
            Attention pooled tensor with shape
                (batch_size, n_queries, output_embed_dim).
        """
        x = self.ln_k(x)
        query = self.ln_q(self.query)
        batch_size = x.shape[0]

        # (n_queries, output_embed_dim) -> (batch_size, n_queries, output_embed_dim)
        query = self._repeat(query, batch_size)

        out = self.attn(query, x, x)
        assert isinstance(out, Tensor)
        out = self.ln_post(out)
        return out

    def _repeat(self, query: Tensor, n: int) -> Tensor:
        return query.unsqueeze(0).repeat(n, 1, 1)


class CascadedAttentionPooler(nn.Module):
    """
    Wrapper class to perform cascaded attention pooling given multiple attention
    poolers. E.g. in CoCa the contrastive pooler is applied on top of the outputs of
    the captioning pooler.

    Args:
        poolers (List[AttentionPooler]): List of individual attention poolers
    """

    def __init__(
        self,
        poolers: List[AttentionPooler],
    ):
        super().__init__()
        self.poolers = nn.ModuleList(poolers)

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_embed_dim).
        Returns:
            List[Tensor] containing attention pooled tensors with shapes
                (batch_size, n_queries, output_embed_dim), where n_queries and
                output_embed_dim are determined by each individual pooler.
        """
        pooler_outs = []
        for pooler in self.poolers:
            x = pooler(x)
            pooler_outs.append(x)
        return pooler_outs
