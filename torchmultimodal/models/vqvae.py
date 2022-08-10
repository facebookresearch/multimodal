# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

from torch import nn, Tensor
from torchmultimodal.modules.layers.codebook import Codebook, CodebookOutput


class VQVAEOutput(NamedTuple):
    decoded: Tensor  # output of decoder
    codebook_output: CodebookOutput  # output of codebook layer to be used in loss calculations


class VQVAE(nn.Module):
    """General model for VQVAE that provides codebook layer to link user specified
    encoder and decoder.

    Vector Quantized Variational Autoencoder is a type of autoencoder that defines
    an embedding of discrete vectors as the latent variables in the bottleneck layer
    instead of normally distributed latent variables as in a standard VAE. This enables
    high-fidelity reconstruction of input data. It was first introduced in "Neural
    Discrete Representation Learning" (Oord et al. 2017) and has since seen success in
    tokenizing and generating high-resolution image, audio, and video data.

    Attributes:
        input_shape (Tuple[int, ...]): Shape of data to be encoded ``[b, c, d1. ,,,. dn]``.
        encoder (nn.Module): Model that accepts single Tensor as input in forward, ``encoder(x)``.
                             Will be used to project input into codebook layer. Expects channel
                             dim of encoder output to match ``codebook_embedding_dim``.
        decoder (nn.Module): Model that accepts single Tensor as input in forward, ``decoder(x)``.
                             Should be able to accept output shape of codebook layer, which matches
                             output shape of encoder.
        codebook_num_embeddings (int): Number of embedding vectors in codebook
        codebook_embedding_dim (int): Dimensionality of embedding vectors in codebook

    Args:
        x (Tensor): Input data of shape ``[b, c, d1, ..., dn]``.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_num_embeddings: int,
        codebook_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(codebook_num_embeddings, codebook_embedding_dim)
        self.latent_shape = self.encoder.get_latent_shape(input_shape)

    def _reshape(self, quantized_flat: Tensor) -> Tensor:
        b, _, emb_dim = quantized_flat.shape
        shape = (b,) + latent_shape + (emb_dim,)
        quantized_flat = quantized_flat.view(shape)  # (b, *latent_shape, emb_dim)
        quantized = shift_dim(quantized_flat, -1, 1)  # (b, emb_dim, *latent_shape)
        return quantized

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        codebook_output = self.codebook(self.encoder(x))
        return codebook_output.codebook_indices, codebook_output.quantized_flat

    def decode(self, token_ids: Tensor) -> Tensor:
        quantized_flat = self.lookup(token_ids)  # (b, seq_len, emb_dim)
        quantized = self._reshape(quantized_flat)  # (b, emb_dim, latent_shape)
        return self.decoder(quantized)  # (b, c, input_shape)

    def lookup(self, token_ids: Tensor) -> Tensor:
        return self.codebook.lookup(token_ids)

    def forward(self, x: Tensor) -> VQVAEOutput:
        quantized_flat = self.codebook(self.encoder(x)).quantized_flat
        quantized = self._reshape(quantized_flat)
        decoded = self.decoder(quantized)
        return VQVAEOutput(decoded, quantized)
