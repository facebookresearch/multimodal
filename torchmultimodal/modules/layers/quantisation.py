# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor


class Quantisation(nn.Module):
    """Quantisation provides an embedding layer that takes in a collection of flattened vectors, usually the
    output of an encoder architecture, and performs a nearest-neighbor lookup in the embedding space.

    Vector quantisation was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf) to generate high-fidelity
    images, videos, and audio data.

    Args:
        num_embeddings (int): the number of vectors in the embedding space
        embedding_dim (int): the dimensionality of the embedding vectors
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )
        self.quantised_vectors = None

    def forward(self, x: Tensor):
        # Rearrange from batch x channel x n dims to batch x n dims x channel
        new_dims = (0,) + tuple(range(2, len(x.shape))) + (1,)
        x_permuted = x.permute(new_dims).contiguous()
        permuted_shape = x_permuted.shape

        # Flatten input
        x_flat = x_permuted.view(-1, permuted_shape[-1])
        # channel dimension should be embedding dim so that each element in encoder
        # output volume gets associated with single embedding vector
        assert (
            x_flat.shape[-1] == self.embedding_dim
        ), f"Expected {x_flat.shape[-1]} to be embedding size of {self.embedding_dim}"

        # Calculate distances from each encoder output vector to each embedding vector, ||x - emb||^2
        distances = torch.cdist(x_flat, self.embedding.weight, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantise
        quantised_permuted = self.embedding(encoding_indices).view(permuted_shape)

        # Straight through estimator
        quantised_permuted = x_permuted + (quantised_permuted - x_permuted).detach()

        # Rearrange back to batch x channel x n dims
        old_dims = (0,) + (len(x.shape) - 1,) + tuple(range(1, len(x.shape) - 1))
        quantised = quantised_permuted.permute(old_dims).contiguous()
        self.quantised_vectors = quantised

        return quantised

    def get_quantised_vectors(self):
        # Retrieve the previously quantised vectors without forward passing again
        if self.quantised_vectors is None:
            raise Exception(
                "quantisation has not yet been performed, please run a forward pass"
            )
        return self.quantised_vectors
