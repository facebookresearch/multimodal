# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class Quantisation(nn.Module):
    """
    Embedding layer that takes in a collection of flattened vectors and finds closest embedding vectors
    to each flattened vector and outputs those selected embedding vectors. Also known as vector quantisation.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.quantised_vectors = None

    def forward(self, x_flat):
        x_shape = x_flat.shape
        # channel dimension should be embedding dim so that each element in encoder
        # output volume gets associated with single embedding vector
        assert (
            x_shape[-1] == self.embedding_dim
        ), f"Expected {x_shape[-1]} to be embedding size of {self.embedding_dim}"

        # Calculate distances from each encoder output vector to each embedding vector, ||x - emb||^2
        w_t = self.embedding.weight.t()
        distances = (
            torch.sum(x_flat ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(x_flat, w_t)
            + torch.sum(w_t ** 2, dim=0, keepdim=True)
        )

        # Encoding - select closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=x_flat.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantise
        quantised_flattened = torch.matmul(encodings, self.embedding.weight)
        self.quantised_vectors = quantised_flattened

        return quantised_flattened

    def get_quantised_vectors(self):
        # Retrieve the previously quantised vectors without forward passing again
        if self.quantised_vectors is None:
            raise Exception(
                "quantisation has not yet been performed, please run a forward pass"
            )
        return self.quantised_vectors
