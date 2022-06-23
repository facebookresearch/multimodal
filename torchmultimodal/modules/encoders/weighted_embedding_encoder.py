# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple, Union

import torch
from torch import nn, Tensor


class WeightedEmbeddingEncoder(nn.Module):
    """Combine weighted embeddings for tensor representing list of indices based on
    pooling type.

    Args:
        embedding (nn.Embedding): embedding module
        pooling_function (Callable[[Tensor, int], Union[Tensor, Tuple]]): pooling function to combine the weighted embeddings,\
        example: torch.sum function should return a tensor or namedtuple containing the tensor in the values field like torch.max
        pooling_dim (int) : dimension along which the pooling function is applied

    Inputs:
        weights (Tensor): A float tensor of shape [batch_size x num_categories] containing the weights of a categorical feature.\
            The weights represent multiplier factors for the corresponding category embedding vectors.

    """

    def __init__(
        self,
        embedding: nn.Embedding,
        pooling_function: Callable[[Tensor, int], Union[Tensor, Tuple]],
        pooling_dim: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.pooling_function = pooling_function
        self.pooling_dim = pooling_dim

    def forward(self, weights: Tensor) -> Tensor:
        index = torch.arange(0, weights.size(1), dtype=torch.int)
        index = index.to(weights.device)
        weighted_embeddings = self.embedding(index) * weights.unsqueeze(-1)
        pooled_embeddings = self.pooling_function(weighted_embeddings, self.pooling_dim)
        if isinstance(pooled_embeddings, Tensor):
            output: Tensor = pooled_embeddings
        else:
            assert hasattr(
                pooled_embeddings, "values"
            ), "pooled embeddings should be Tensor or tuple with values field as Tensor"
            output = pooled_embeddings.values  # type: ignore
        return output
