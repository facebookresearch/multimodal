# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Tuple

import torch
from torch import nn, Size, Tensor


class QuantizationOutput(NamedTuple):
    encoded_flat: Tensor  # the flattened encoder output
    quantized_flat: Tensor  # the chosen nearest embeddings
    codebook_indices: Tensor  # indices of the chosen embeddings
    quantized: Tensor  # the chosen embeddings (unflattened)


class Quantization(nn.Module):
    """Quantization provides an embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.

    Vector quantization was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
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

    def _preprocess(self, encoded: Tensor) -> Tuple[Tensor, Size]:
        # Rearrange from batch x channel x n dims to batch x n dims x channel
        new_dims = (0,) + tuple(range(2, len(encoded.shape))) + (1,)
        encoded_permuted = encoded.permute(new_dims).contiguous()
        permuted_shape = encoded_permuted.shape

        # Flatten input
        encoded_flat = encoded_permuted.view(-1, permuted_shape[-1])
        # channel dimension should be embedding dim so that each element in encoder
        # output volume gets associated with single embedding vector
        if encoded_flat.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected {encoded_flat.shape[-1]} to be embedding size of {self.embedding_dim}"
            )

        return encoded_flat, permuted_shape

    def _postprocess(self, quantized_flat: Tensor, permuted_shape: Size) -> Tensor:
        # Rearrange back to batch x channel x n dims
        num_dims = len(permuted_shape)
        quantized_permuted = quantized_flat.view(permuted_shape)
        old_dims = (0,) + (num_dims - 1,) + tuple(range(1, num_dims - 1))
        quantized = quantized_permuted.permute(old_dims).contiguous()

        return quantized

    def _ema_update_embedding(self, encoded_flat: Tensor, codebook_indices: Tensor):
        # Closed form solution of codebook loss, ||e - E(x)||^2, is simply the average
        # of the encoder output. However, we can't compute this in minibatches, so we
        # must use exponential moving average.

        # Convert indices to one hot encoding
        codebook_onehot = nn.functional.one_hot(
            codebook_indices, num_classes=self.num_embeddings
        )
        # Count how often each embedding vector was looked up
        codebook_selection_count = torch.sum(codebook_onehot, 0)
        # Update usage value for each embedding vector
        self._code_usage = self._code_usage * self._decay + codebook_selection_count * (
            1 - self._decay
        )
        # Laplace smoothing of codebook usage - to prevent zero counts
        n = torch.sum(self._code_usage)
        self._code_usage = (
            (self._code_usage + self._epsilon)
            / (n + self.num_embeddings * self._epsilon)
            * n
        )
        # Get all encoded vectors attracted to each embedding vector
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), encoded_flat)
        # Update each embedding vector with the sum of encoded vectors that are attracted to it,
        # divided by its usage to yield the mean of encoded vectors that choose it
        self._code_sum = nn.Parameter(
            self._code_sum * self._decay + (1 - self._decay) * encoded_per_codebook
        )
        self.embedding.weight = nn.Parameter(
            self._code_sum / self._code_usage.unsqueeze(1)
        )

    def _quantize(self, encoded_flat: Tensor) -> Tuple[Tensor, Tensor]:
        # Calculate distances from each encoder, E(x), output vector to each embedding vector, e, ||E(x) - e||^2
        distances = torch.cdist(encoded_flat, self.embedding.weight, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        codebook_indices = torch.argmin(distances, dim=1)

        # Quantize
        quantized_flat = self.embedding(codebook_indices)

        # Use exponential moving average to update the embedding instead of a codebook loss,
        # as suggested by Oord et al. 2017 and Razavi et al. 2019.
        if self.training:
            self._ema_update_embedding(encoded_flat, codebook_indices)

        # Straight through estimator
        quantized_flat = encoded_flat + (quantized_flat - encoded_flat).detach()

        return quantized_flat, codebook_indices

    def forward(self, x: Tensor) -> QuantizationOutput:
        # Reshape and flatten encoder output for quantization
        encoded_flat, permuted_shape = self._preprocess(x)

        # Quantization via nearest neighbor lookup
        quantized_flat, codebook_indices = self._quantize(encoded_flat)

        # Reshape back to original dims
        quantized = self._postprocess(quantized_flat, permuted_shape)

        return QuantizationOutput(
            encoded_flat, quantized_flat, codebook_indices, quantized
        )
