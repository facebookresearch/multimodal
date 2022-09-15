# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, NamedTuple, Tuple, Union

import torch
from torch import nn, Size, Tensor
from torch.nn import functional as F
from torchmultimodal.utils.common import shift_dim


class CodebookOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.modules.layers.codebook.Codebook`.

    Attributes:
        encoded_flat (Tensor): The flattened encoder output of shape ``(b x d1 x ... x dn, c)``.
        quantized_flat (Tensor): The nearest embeddings for the encoded of shape ``(b x d1 x ... x dn, emb_dim)``.
        codebook_indices (Tensor): Indices of the nearest embeddings of shape ``(b, d1, d2, ..., dn)``.
        quantized (Tensor): The nearest embeddings reshaped back to ``(b, emb_dim, d1, ..., dn)``.
    """

    encoded_flat: Tensor
    quantized_flat: Tensor
    codebook_indices: Tensor
    quantized: Tensor


class Codebook(nn.Module):
    """Bottleneck layer of VQVAE model

    Codebook provides an embedding layer that takes in the output of an encoder
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
        num_embeddings (int): Number of vectors in the embedding space.
        embedding_dim (int): Dimensionality of the embedding vectors.
        decay (float, optional): Factor used in exponential moving average update of the embeddings.
            Defaults to ``0.99``.
        codebook_usage_threshold (float, optional): Threshold for the average number of times an embedding vector
            is chosen below which it will be re-initialized. Defaults to ``1.0``.
        epsilon (float, optional): Noise used in Laplace smoothing of codebook usage. Defaults to ``1e-7``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        codebook_usage_threshold: float = 1.0,
        epsilon: float = 1e-7,
    ) -> None:
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

        # Threshold for randomly reseting unused embedding vectors
        self.codebook_usage_threshold = codebook_usage_threshold

        # Flag to track if we need to initialize embedding with encoder output
        self._is_embedding_init = False

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        # Override nn.Module's _load_from_state_dict to ensure embedding init is turned off
        # when state dict is loaded.
        #
        # This can also be handled with _register_load_state_dict_pre_hook but since this is
        # an internal function, it may change. Overriding _load_from_state_dict seems more
        # stable and cleaner.
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._is_embedding_init = True

    def _tile(self, x: Tensor, n: int) -> Tensor:
        # Repeat vectors in x if x has less than n vectors
        num_vectors, num_channels = x.shape
        if num_vectors < n:
            num_repeats = (n + num_vectors - 1) // num_vectors
            # Add a small amount of noise to repeated vectors
            std = 0.01 / torch.sqrt(torch.tensor(num_channels))
            x = x.repeat(num_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _get_random_vectors(self, x: Tensor, n: int) -> Tensor:
        # Gets n random row vectors from 2D tensor x
        x_tiled = self._tile(x, n)
        idx = torch.randperm(x_tiled.shape[0])
        x_rand = x_tiled[idx][:n]
        return x_rand

    def _preprocess(self, encoded: Tensor) -> Tuple[Tensor, Size]:
        # Rearrange from batch x channel x n dims to batch x n dims x channel
        encoded_permuted = shift_dim(encoded, 1, -1)
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

    def _postprocess(
        self, quantized_flat: Tensor, permuted_shape: Union[Size, Tuple]
    ) -> Tensor:
        # Rearrange back to batch x channel x n dims
        quantized_permuted = quantized_flat.view(permuted_shape)
        quantized = shift_dim(quantized_permuted, -1, 1)

        return quantized

    def _init_embedding(self, encoded_flat: Tensor) -> None:
        # Embedding should be initialized with random output vectors from the encoder
        # on the first forward pass for faster convergence, as in VideoGPT (Yan et al. 2021)
        #
        # This requires the preprocessed encoder output to flattened

        self._is_embedding_init = True

        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)

        # Initialize embedding and intermediate values for EMA updates
        self.embedding = encoded_flat_rand
        self.code_avg = encoded_flat_rand
        self.code_usage = torch.ones(self.num_embeddings)

    def _ema_update_embedding(
        self, encoded_flat: Tensor, codebook_indices: Tensor
    ) -> None:
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
        self.code_usage.mul_(self._decay).add_(
            codebook_selection_count, alpha=(1 - self._decay)
        )
        # Laplace smoothing of codebook usage - to prevent zero counts
        n = torch.sum(self.code_usage)
        self.code_usage.add_(self._epsilon).divide_(
            n + self.num_embeddings * self._epsilon
        ).mul_(n)
        # Get all encoded vectors attracted to each embedding vector
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), encoded_flat)
        # Update each embedding vector with new encoded vectors that are attracted to it,
        # divided by its usage to yield the mean of encoded vectors that choose it
        self.code_avg.mul_(self._decay).add_(
            encoded_per_codebook, alpha=(1 - self._decay)
        )
        self.embedding = self.code_avg / self.code_usage.unsqueeze(1)
        # Reset any embedding vectors that fall below threshold usage with random encoded vectors
        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)
        self.embedding = torch.where(
            self.code_usage.unsqueeze(1) >= self.codebook_usage_threshold,
            self.embedding,
            encoded_flat_rand,
        )

    def _quantize(self, encoded_flat: Tensor) -> Tuple[Tensor, Tensor]:
        # Calculate distances from each encoder, E(x), output vector to each embedding vector, e, ||E(x) - e||^2
        distances = torch.cdist(encoded_flat, self.embedding, p=2.0) ** 2

        # Encoding - select closest embedding vectors
        codebook_indices_flat = torch.argmin(distances, dim=1)

        # Quantize
        quantized_flat = F.embedding(codebook_indices_flat, self.embedding)

        # Use exponential moving average to update the embedding instead of a codebook loss,
        # as suggested by Oord et al. 2017 and Razavi et al. 2019.
        if self.training:
            self._ema_update_embedding(encoded_flat, codebook_indices_flat)

        # Straight through estimator
        quantized_flat = encoded_flat + (quantized_flat - encoded_flat).detach()

        return quantized_flat, codebook_indices_flat

    def forward(self, z: Tensor) -> CodebookOutput:
        """
        Args:
            z (Tensor): Tensor containing a batch of encoder outputs of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.modules.layers.codebook.CodebookOutput`.
        """
        # Flatten encoder outputs, tile to match num embeddings, get random encoder outputs
        encoded_flat, permuted_shape = self._preprocess(z)

        # First check if embedding is initialized correctly
        if not self._is_embedding_init and self.training:
            self._init_embedding(encoded_flat)

        # Quantization via nearest neighbor lookup
        quantized_flat, codebook_indices_flat = self._quantize(
            encoded_flat
        )  # (b x d1 x ... x dn, emb_dim)

        # Reshape back to original dims
        # Note: This part could also happen before ema_update_embedding by first reshaping the indices
        # and then looking up the codebook for quantized. But that will require us to pass shape info
        # into `self._quantized`. We decide to keep the reshape and the quantized ops separate for clarity.
        quantized = self._postprocess(
            quantized_flat, permuted_shape
        )  # (b, emb_dim, d1, ...., dn)
        codebook_indices = codebook_indices_flat.view(
            z.shape[0], *z.shape[2:]
        )  # (b, d1, ..., dn)

        return CodebookOutput(encoded_flat, quantized_flat, codebook_indices, quantized)

    def extra_repr(self) -> str:
        return "num_embeddings={}, embedding_dim={}".format(
            self.num_embeddings, self.embedding_dim
        )

    def lookup(self, indices: Tensor) -> Tensor:
        # Returns the embeddings of shape ``[b, indices.shape, emb_dim]``
        return F.embedding(indices, self.embedding)
