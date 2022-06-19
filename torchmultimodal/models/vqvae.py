# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

from torch import nn, Tensor
from torchmultimodal.modules.layers import Codebook
from torchmultimodal.modules.layers.codebook import CodebookOutput


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

    Args:
        encoder (nn.Module): model that accepts single Tensor as input in forward, ``encoder(x)``.
                             Will be used to project input into codebook layer. Expects channel
                             dim of encoder output to match ``codebook_embedding_dim``.
        decoder (nn.Module): model that accepts single Tensor as input in forward, ``decoder(x)``.
                             Should be able to accept output shape of codebook layer, which matches
                             output shape of encoder.
        codebook_num_embeddings (int): number of embedding vectors in codebook
        codebook_embedding_dim (int): dimensionality of embedding vectors in codebook

    Inputs:
        x (Tensor): [b, c, d1, ..., dn] tensor
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        codebook_num_embeddings: int,
        codebook_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(codebook_num_embeddings, codebook_embedding_dim)

    def encode(self, x: Tensor) -> CodebookOutput:
        return self.codebook(self.encoder(x))

    def decode(self, e: Tensor) -> Tensor:
        return self.decoder(e)

    def tokenize(self, x: Tensor) -> Tensor:
        """Similar to encode, but return flattened quantized outputs"""
        quantized = self.encode(x)
        return quantized.quantized_flat

    def forward(self, x: Tensor) -> VQVAEOutput:
        quantized = self.encode(x)
        decoded = self.decode(quantized.quantized)
        return VQVAEOutput(decoded, quantized)
