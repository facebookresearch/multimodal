# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Tuple

import torch
from torch import nn, Size, Tensor


class CodebookOutput(NamedTuple):
    encoded_flat: Tensor  # the flattened encoder output
    quantized_flat: Tensor  # the chosen nearest embeddings
    codebook_indices: Tensor  # indices of the chosen embeddings
    quantized: Tensor  # the chosen embeddings (unflattened)


class Codebook(nn.Module):
    """Codebook provides an embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.

    Vector quantization was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
    to generate high-fidelity images, videos, and audio data.

    The embedding weights are trained with exponential moving average updates as described
    in original paper.

    Code was largely inspired by a PyTorch implementation of the author's original code, found here:
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    and by the implementation in MUGEN (Hayes et al. 2022), found here:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/video_vqvae/vqvae.py

    Args:
        num_embeddings (int): the number of vectors in the embedding space
        embedding_dim (int): the dimensionality of the embedding vectors

    Inputs:
        z (Tensor): Tensor containing a batch of encoder outputs.
                    Expects dimensions to be batch x channel x n dims.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        # Embedding weights and parameters for EMA update will be registered to buffer, as they
        # will not be updated by the optimizer but are still model parameters.
        # code_usage and code_avg correspond with N and m, respectively, from Oord et al.
        randn_init_embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", randn_init_embedding.clone())
        self.register_buffer("code_usage", torch.zeros(num_embeddings))
        self.register_buffer("code_avg", randn_init_embedding.clone())

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self._decay = decay
        # Used in Laplace smoothing of code usage
        self._epsilon = epsilon

        # Flag to track if we need to initialize embedding with encoder output
        self._is_embedding_init = False

    def _tile(self, x):
        # Repeat encoder vectors in cases where the encoder output does not have enough vectors
        # to initialize the codebook on first forward pass
        num_encoder_vectors, num_channels = x.shape
        if num_encoder_vectors < self.embedding_dim:
            num_repeats = (
                self.num_embeddings + num_encoder_vectors - 1
            ) // num_encoder_vectors
            # Add a small amount of noise to repeated vectors
            std = 0.01 / torch.sqrt(torch.tensor(num_channels))
            x = x.repeat(num_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

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

    def _init_embedding_and_preprocess(self, z: Tensor) -> Tuple[Tensor, Size]:
        # Embedding should be initialized with random output vectors from the encoder
        # on the first forward pass for faster convergence, as in VideoGPT (Yan et al. 2021)
        #
        # This requires preprocessing the encoder output, so return this as well.

        self._is_embedding_init = True

        # Flatten encoder outputs, tile to match num embeddings, get random encoder outputs
        encoded_flat, permuted_shape = self._preprocess(z)
        encoded_flat_tiled = self._tile(encoded_flat)
        idx = torch.randperm(encoded_flat_tiled.shape[0])
        encoded_flat_rand = encoded_flat_tiled[idx][: self.num_embeddings]

        # Initialize embedding and intermediate values for EMA updates
        self.embedding = encoded_flat_rand
        self.code_avg = encoded_flat_rand
        self.code_usage = torch.ones(self.num_embeddings)

        return encoded_flat, permuted_shape

    def _ema_update_embedding(self, encoded_flat: Tensor, codebook_indices: Tensor):
        # Closed form solution of codebook loss, ||e - E(x)||^2, is simply the average
        # of the encoder output. However, we can't compute this in minibatches, so we
        # must use exponential moving average.

        # Convert indices to one hot encoding
        codebook_onehot = nn.functional.one_hot(
            codebook_indices, num_classes=self.num_embeddings
        ).type(torch.float)
        # Count how often each embedding vector was looked up
        codebook_selection_count = torch.sum(codebook_onehot, 0)
        # Update usage value for each embedding vector
        self.code_usage = self.code_usage * self._decay + codebook_selection_count * (
            1 - self._decay
        )
        # Laplace smoothing of codebook usage - to prevent zero counts
        n = torch.sum(self.code_usage)
        self.code_usage = (
            (self.code_usage + self._epsilon)
            / (n + self.num_embeddings * self._epsilon)
            * n
        )
        # Get all encoded vectors attracted to each embedding vector
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), encoded_flat)
        # Update each embedding vector with new encoded vectors that are attracted to it,
        # divided by its usage to yield the mean of encoded vectors that choose it
        self.code_avg = (
            self.code_avg * self._decay + (1 - self._decay) * encoded_per_codebook
        )
        self.embedding = self.code_avg / self.code_usage.unsqueeze(1)

    def _quantize(self, encoded_flat: Tensor) -> Tuple[Tensor, Tensor]:
        # Calculate distances from each encoder, E(x), output vector to each embedding vector, e, ||E(x) - e||^2
        distances = torch.cdist(encoded_flat, self.embedding, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        codebook_indices = torch.argmin(distances, dim=1)

        # Quantize
        quantized_flat = self.embedding[codebook_indices]

        # Use exponential moving average to update the embedding instead of a codebook loss,
        # as suggested by Oord et al. 2017 and Razavi et al. 2019.
        if self.training:
            self._ema_update_embedding(encoded_flat, codebook_indices)

        # Straight through estimator
        quantized_flat = encoded_flat + (quantized_flat - encoded_flat).detach()

        return quantized_flat, codebook_indices

    def forward(self, z: Tensor) -> CodebookOutput:
        # First check if embedding is initialized correctly
        if not self._is_embedding_init and self.training:
            encoded_flat, permuted_shape = self._init_embedding_and_preprocess(z)
        else:
            # Reshape and flatten encoder output for quantization
            encoded_flat, permuted_shape = self._preprocess(z)

        # Quantization via nearest neighbor lookup
        quantized_flat, codebook_indices = self._quantize(encoded_flat)

        # Reshape back to original dims
        quantized = self._postprocess(quantized_flat, permuted_shape)

        return CodebookOutput(encoded_flat, quantized_flat, codebook_indices, quantized)
