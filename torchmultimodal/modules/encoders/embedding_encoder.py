# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor


POOLING_TYPES = ["sum", "mean", "max"]


class EmbeddingEncoder(nn.Module):
    """Combine embeddings for tensor representing list of indices based on pooling type

    Args:
        embedding (nn.Embedding): embedding module
        pooling_type (str): pooling function to combine the embeddings like sum. Choose
        from pooling_types
        pooling_dim (int) : dimension along which the pooling function is applied
        use_hash (bool): if hashing based on embedding vocab size if applied to input
        before embedding layer

    Inputs:
        x (Tensor): Tensor bsz x max seq length representing (padded) list of indices
        for embedding

    """

    def __init__(
        self,
        embedding: nn.Embedding,
        pooling_type: str,
        pooling_dim: int = 1,
        use_hash: bool = False,
    ):
        super().__init__()
        self.embedding = embedding
        if pooling_type not in POOLING_TYPES:
            raise ValueError(
                f"pooling type should be in {POOLING_TYPES}, found {pooling_type}"
            )
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim
        self.use_hash = use_hash

    def forward(self, x: Tensor) -> Tensor:
        if self.use_hash:
            x = x % (self.embedding.num_embeddings - 1) + 1
        out = self.embedding(x)
        if self.pooling_type == "sum":
            out = torch.sum(out, dim=self.pooling_dim)
        elif self.pooling_type == "mean":
            out = torch.mean(out, dim=self.pooling_dim)
        else:
            out = torch.max(out, dim=self.pooling_dim).values
        return out
