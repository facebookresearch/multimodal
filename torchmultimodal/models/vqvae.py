# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple, Tuple, Union

from torch import nn, Size, Tensor
from torchmultimodal.modules.layers.codebook import Codebook, CodebookOutput
from torchmultimodal.utils.common import shift_dim


class VQVAEOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.vqvae.VQVAE`.

    Attributes:
        decoded (Tensor): Output of the decoder.
        codebook_output (CodebookOutput): Output of codebook layer to be used in loss calculations.
    """

    decoded: Tensor
    codebook_output: CodebookOutput


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
        encoder (nn.Module): Model that accepts single Tensor as input in forward, ``encoder(x)``.
            Will be used to project input into codebook layer. Expects channel
            dim of encoder output to match ``embedding_dim`` of codebook.
            See :class:`~torchmultimodal.modules.layers.codebook.Codebook`.
        decoder (nn.Module): Model that accepts single Tensor as input in forward, ``decoder(x)``.
            Should be able to accept output shape of codebook layer, which matches output shape of
            the encoder.
        num_embeddings (int): Number of embedding vectors in codebook.
        embedding_dim (int): Dimensionality of embedding vectors in codebook.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def latent_shape(self, input_shape: Union[Size, Tuple]) -> Tuple[int, ...]:
        """Returns the downsampled shape of the encoder output: (d1, ..., dn)"""
        if not hasattr(self.encoder, "get_latent_shape"):
            raise AttributeError(
                f"Missing attribute 'get_latent_shape' of the encoder {self.encoder}"
            )

        return self.encoder.get_latent_shape(input_shape)  # type: ignore

    def encode(
        self, x: Tensor, return_embeddings: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """Converts input data to token ids

        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.
            return_embeddings (bool): Flag to return also the quantized embeddings. Defaults to ``False``.

        Returns:
            * A tensor of token ids: ``(b, d1, ...., dn)``
            * A tuple of token ids and quantized embeddings ``(b, emb_dim, d1, ..., dn)``.
        """
        encoded = self.encoder(x)
        out = self.codebook(encoded)
        indices = out.codebook_indices
        quantized = out.quantized
        if return_embeddings:
            return indices, quantized
        return indices

    def decode(self, indices: Tensor) -> Tensor:
        """Converts token ids back to data"""
        quantized = self.lookup(indices)  # (b, latent_shape, emb_dim)
        quantized = shift_dim(quantized, -1, 1)  # (b, emb_dim, latent_shape)
        return self.decoder(quantized)  # (b, c, input_shape)

    def lookup(self, indices: Tensor) -> Tensor:
        if not hasattr(self.codebook, "lookup"):
            raise AttributeError(
                f"Missing attribute 'lookup' of the codebook {self.codebook}"
            )

        return self.codebook.lookup(indices)

    def forward(self, x: Tensor) -> VQVAEOutput:
        """
        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.models.vqvae.VQVAEOutput`.
        """
        encoded = self.encoder(x)
        codebook_output = self.codebook(encoded)
        decoded = self.decoder(codebook_output.quantized)
        return VQVAEOutput(decoded, codebook_output)
