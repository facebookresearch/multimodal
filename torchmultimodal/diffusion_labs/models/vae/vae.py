# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import NamedTuple, Sequence

import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Normal
from torchmultimodal.diffusion_labs.models.vae.encoder_decoder import (
    ResNetDecoder,
    ResNetEncoder,
)


class VAEOutput(NamedTuple):
    posterior: Distribution
    decoder_output: Tensor


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (https://arxiv.org/abs/1906.02691) is a special type of autoencoder
    where the encoder outputs the the parameters of the posterior latent distribution instead of
    outputting fixed vectors in the latent space. The decoder consumes a sample from the latent
    distribution to reconstruct the inputs.

    Follows the architecture used in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py#L285

    Attributes:
        encoder (nn.Module): instance of encoder module.
        decoder (nn.Module): instance of decoder module.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        sample_posterior (bool): if True, sample from posterior instead of distribution mpde.
            Defaults to True.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor) -> Distribution:
        h = self.encoder(x)
        # output of encoder is mean and log variaance of a normal distribution
        mean, log_variance = torch.chunk(h, 2, dim=1)
        # clamp logvariance to [-30. 20]
        log_variance = torch.clamp(log_variance, -30.0, 20.0)
        stddev = torch.exp(log_variance / 2.0)
        posterior = Normal(mean, stddev)
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor, sample_posterior: bool = True) -> VAEOutput:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.rsample()
        else:
            z = posterior.mode
        decoder_out = self.decode(z)
        return VAEOutput(posterior=posterior, decoder_output=decoder_out)


def ldm_variational_autoencoder(
    *,
    embedding_channels: int,
    in_channels: int,
    out_channels: int,
    z_channels: int,
    channels: int,
    num_res_blocks: int,
    channel_multipliers: Sequence[int] = (1, 2, 4, 8),
    dropout: float = 0.0,
    norm_groups: int = 32,
    norm_eps: float = 1e-6,
    output_alpha_channel: bool = False,
) -> VariationalAutoencoder:
    encoder = nn.Sequential(
        # pyre-ignore
        OrderedDict(
            [
                (
                    "resnet_encoder",
                    ResNetEncoder(
                        in_channels=in_channels,
                        z_channels=z_channels,
                        channels=channels,
                        num_res_blocks=num_res_blocks,
                        channel_multipliers=channel_multipliers,
                        dropout=dropout,
                        norm_groups=norm_groups,
                        norm_eps=norm_eps,
                        double_z=True,
                    ),
                ),
                (
                    "quant_conv",
                    nn.Conv2d(2 * z_channels, 2 * embedding_channels, kernel_size=1),
                ),
            ]
        )
    )

    decoder = nn.Sequential(
        # pyre-ignore
        OrderedDict(
            [
                (
                    "post_quant_conv",
                    nn.Conv2d(embedding_channels, z_channels, kernel_size=1),
                ),
                (
                    "resnet_decoder",
                    ResNetDecoder(
                        out_channels=out_channels,
                        z_channels=z_channels,
                        channels=channels,
                        num_res_blocks=num_res_blocks,
                        channel_multipliers=channel_multipliers,
                        dropout=dropout,
                        norm_groups=norm_groups,
                        norm_eps=norm_eps,
                        output_alpha_channel=output_alpha_channel,
                    ),
                ),
            ]
        )
    )

    return VariationalAutoencoder(encoder=encoder, decoder=decoder)
