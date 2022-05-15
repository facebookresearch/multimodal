# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Size, Tensor


class Quantisation(nn.Module):
    """Quantisation provides an embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.

    Vector quantisation was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
    to generate high-fidelity images, videos, and audio data.

    Args:
        num_embeddings (int): the number of vectors in the embedding space
        embedding_dim (int): the dimensionality of the embedding vectors

    Inputs:
        x (Tensor): Tensor containing a batch of encoder outputs.
                    Expects dimensions to be batch x channel x n dims.
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

    def _preprocess(self, x: Tensor):
        # Rearrange from batch x channel x n dims to batch x n dims x channel
        new_dims = (0,) + tuple(range(2, len(x.shape))) + (1,)
        x_permuted = x.permute(new_dims).contiguous()
        permuted_shape = x_permuted.shape

        # Flatten input
        x_flat = x_permuted.view(-1, permuted_shape[-1])
        # channel dimension should be embedding dim so that each element in encoder
        # output volume gets associated with single embedding vector
        if x_flat.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected {x_flat.shape[-1]} to be embedding size of {self.embedding_dim}"
            )

        return x_flat, permuted_shape

    def _postprocess(self, quantised_flat: Tensor, permuted_shape: Size):
        # Rearrange back to batch x channel x n dims
        num_dims = len(permuted_shape)
        quantised_permuted = quantised_flat.view(permuted_shape)
        old_dims = (0,) + (num_dims - 1,) + tuple(range(1, num_dims - 1))
        quantised = quantised_permuted.permute(old_dims).contiguous()

        return quantised

    def quantise(self, x_flat: Tensor):
        # Calculate distances from each encoder output vector to each embedding vector, ||x - emb||^2
        distances = torch.cdist(x_flat, self.embedding.weight, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantise
        quantised_flat = self.embedding(encoding_indices)

        # Straight through estimator
        quantised_flat = x_flat + (quantised_flat - x_flat).detach()

        return quantised_flat

    def forward(self, x: Tensor):
        # Reshape and flatten encoder output for quantisation
        x_flat, permuted_shape = self._preprocess(x)

        # Quantisation via nearest neighbor lookup
        quantised_flat = self.quantise(x_flat)

        # Reshape back to original dims
        quantised = self._postprocess(quantised_flat, permuted_shape)
        self.quantised_vectors = quantised

        return quantised

    def get_quantised_vectors(self):
        # Retrieve the previously quantised vectors without forward passing again
        if self.quantised_vectors is None:
            raise Exception(
                "quantisation has not yet been performed, please run a forward pass"
            )
        return self.quantised_vectors
