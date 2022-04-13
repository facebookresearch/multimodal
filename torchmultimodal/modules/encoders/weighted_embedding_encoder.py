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
        use_hash (bool): if hashing based on embedding vocab size if applied to input
        before embedding layer

    Inputs:
        x (Tensor): Tensor bsz x len where first half (0 : len/2) of tensor are embedding indices
        and the second half are corresponding weights for the embedding indices

    """

    def __init__(
        self,
        embedding: nn.Embedding,
        pooling_function: Callable[[Tensor, int], Union[Tensor, Tuple]],
        pooling_dim: int = 1,
        use_hash: bool = False,
    ) -> None:
        super().__init__()
        if (
            use_hash
            and embedding.padding_idx is not None
            and embedding.padding_idx != 0
        ):
            raise ValueError("embedding padding should be None or 0 if hashing is used")
        self.embedding = embedding
        self.pooling_function = pooling_function
        self.pooling_dim = pooling_dim
        self.use_hash = use_hash

    def forward(self, x: Tensor) -> Tensor:
        index, weights = torch.split(
            x,
            int(x.size()[1] / 2),
            dim=1,
        )
        index = index.long()
        if self.use_hash:
            # TODO: pull this out into a common function T111523602
            if self.embedding.padding_idx is None:
                index = index % self.embedding.num_embeddings
            else:
                mask = ~index.eq(self.embedding.padding_idx)
                non_zero_index = torch.masked_select(index, mask)
                index[mask] = (non_zero_index - 1) % (
                    self.embedding.num_embeddings - 1
                ) + 1

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
