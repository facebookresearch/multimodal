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

    def forward(self, x_flat: Tensor):
        x_shape = x_flat.shape
        # channel dimension should be embedding dim so that each element in encoder
        # output volume gets associated with single embedding vector
        assert (
            x_shape[-1] == self.embedding_dim
        ), f"Expected {x_shape[-1]} to be embedding size of {self.embedding_dim}"

        # Calculate distances from each encoder output vector to each embedding vector, ||x - emb||^2
        w_t = self.embedding.weight.t()
        distances = torch.cdist(x_flat, w_t, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantise
        quantised_flattened = self.embedding(encoding_indices)

        # Straight through estimator
        quantised_flattened = x_flat + (quantised_flattened - x_flat).detach()
        self.quantised_vectors = quantised_flattened

        return quantised_flattened

    def get_quantised_vectors(self):
        # Retrieve the previously quantised vectors without forward passing again
        if self.quantised_vectors is None:
            raise Exception(
                "quantisation has not yet been performed, please run a forward pass"
            )
        return self.quantised_vectors
