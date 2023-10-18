# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Sequence

import torch
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.models.vae.attention import attention_res_block
from torchmultimodal.diffusion_labs.models.vae.res_block import ResBlock
from torchmultimodal.diffusion_labs.models.vae.residual_sampling import (
    Downsample2D,
    Upsample2D,
)
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm


class ResNetEncoder(nn.Module):
    """Resnet encoder used in the LDM Autoencoder that consists of a init convolution,
    downsampling resnet blocks, middle resnet blocks with attention and output convolution block
    with group normalization and nonlinearity.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#L368

    Attributes:
        in_channels (int): number of input channels.
        z_channels (int): number of latent channels.
        channels (int): number of channels in the initial convolution layer.
        num_res_block (int): number of residual blocks at each resolution.
        channel_multipliers (Sequence[int]): list of channel multipliers. Defaults to [1, 2, 4, 8].
        dropout (float): dropout probability. Defaults to 0.0.
        double_z (bool): whether to use double z_channels for images or not. Defaults to True.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `channels` * `channel_multipliers[-1]` is not divisible by `norm_groups`.
    """

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        channels: int,
        num_res_blocks: int,
        channel_multipliers: Sequence[int] = (
            1,
            2,
            4,
            8,
        ),
        dropout: float = 0.0,
        double_z: bool = True,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        # initial convolution
        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # downsampling block
        self.down_block = nn.Sequential()
        channels_list = tuple(
            [channels * multiplier for multiplier in [1] + list(channel_multipliers)]
        )
        num_resolutions = len(channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = channels_list[level_idx]
            block_out = channels_list[level_idx + 1]
            self.down_block.append(
                res_block_stack(
                    block_in,
                    block_out,
                    num_res_blocks,
                    dropout,
                    needs_downsample=True
                    if level_idx != num_resolutions - 1
                    else False,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
            )

        mid_channels = channels_list[-1]
        self.mid_block = nn.Sequential(
            res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps),
            attention_res_block(mid_channels, norm_groups, norm_eps),
            res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps),
        )

        if mid_channels % norm_groups != 0:
            raise ValueError(
                "Channel dims obtained by multiplying channels with last"
                " item in channel_multipliers needs to be divisible by norm_groups"
            )

        self.out_block = nn.Sequential(
            Fp32GroupNorm(
                num_groups=norm_groups, num_channels=mid_channels, eps=norm_eps
            ),
            nn.SiLU(),
            nn.Conv2d(
                mid_channels,
                out_channels=2 * z_channels if double_z else z_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.init_conv(x)
        h = self.down_block(h)
        h = self.mid_block(h)
        h = self.out_block(h)
        return h


class ResNetDecoder(nn.Module):
    """Resnet decoder used in the LDM Autoencoder that consists of a init convolution,
    middle resnet blocks with attention, upsamling resnet blocks and output convolution
    block with group normalization and nonlinearity. Optionally, also supports alpha
    channel in output.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#L462

    Attributes:
        out_channels (int): number of channels in output image.
        z_channels (int): number of latent channels.
        channels (int): number of channels to be used with channel multipliers.
        num_res_block (int): number of residual blocks at each resolution.
        channel_multipliers (Sequence[int]): list of channel multipliers used by the encoder.
            Decoder uses them in reverse order. Defaults to [1, 2, 4, 8].
        dropout (float): dropout probability. Defaults to 0.0.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.
        output_alpha_channel (bool): whether to include an alpha channel in the output.
            Defaults to False.

    Args:
        z (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `channels` * `channel_multipliers[-1]` is not divisible by `norm_groups`.
    """

    def __init__(
        self,
        out_channels: int,
        z_channels: int,
        channels: int,
        num_res_blocks: int,
        channel_multipliers: Sequence[int] = (
            1,
            2,
            4,
            8,
        ),
        dropout: float = 0.0,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
        output_alpha_channel: bool = False,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.output_alpha_channel = output_alpha_channel

        channels_list = tuple(
            reversed(
                [
                    channels * multiplier
                    for multiplier in list(channel_multipliers)
                    + [channel_multipliers[-1]]
                ]
            )
        )
        mid_channels = channels_list[0]

        # initial convolution
        self.init_conv = nn.Conv2d(z_channels, mid_channels, kernel_size=3, padding=1)

        # middle block
        self.mid_block = nn.Sequential(
            res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps),
            attention_res_block(mid_channels, norm_groups, norm_eps),
            res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps),
        )

        # upsample block
        self.up_block = nn.Sequential()
        num_resolutions = len(channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = channels_list[level_idx]
            block_out = channels_list[level_idx + 1]
            self.up_block.append(
                res_block_stack(
                    block_in,
                    block_out,
                    # decoder creates 1 additional res block compared to encoder.
                    # not sure about intuition, but seems to be used everywhere in OSS.
                    num_res_blocks + 1,
                    dropout,
                    needs_upsample=True if level_idx != num_resolutions - 1 else False,
                    norm_groups=norm_groups,
                    norm_eps=norm_eps,
                )
            )

        # output nonlinearity block
        post_upsample_channels = channels_list[-1]
        if post_upsample_channels % norm_groups != 0:
            raise ValueError(
                "Channel dims obtained by multiplying channels with first"
                " item in channel_multipliers needs to be divisible by norm_groups"
            )
        self.out_nonlinearity_block = nn.Sequential(
            Fp32GroupNorm(
                num_groups=norm_groups,
                num_channels=post_upsample_channels,
                eps=norm_eps,
            ),
            nn.SiLU(),
        )

        # output projections
        self.conv_out = nn.Conv2d(
            post_upsample_channels, out_channels, kernel_size=3, padding=1
        )
        if self.output_alpha_channel:
            self.alpha_conv_out = nn.Conv2d(
                post_upsample_channels, 1, kernel_size=3, padding=1
            )

    def forward(self, z: Tensor) -> Tensor:
        h = self.init_conv(z)
        h = self.mid_block(h)
        h = self.up_block(h)
        h = self.out_nonlinearity_block(h)

        # If alpha channel is required as output, compute it separately with its
        # own conv layer and concatenate with the output from the out convolulution
        if self.output_alpha_channel:
            h = torch.cat((self.conv_out(h), self.alpha_conv_out(h)), dim=1)
        else:
            h = self.conv_out(h)

        return h


def res_block_stack(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    dropout: float = 0.0,
    needs_upsample: bool = False,
    needs_downsample: bool = False,
    norm_groups: int = 32,
    norm_eps: float = 1e-6,
) -> nn.Module:
    if needs_upsample and needs_downsample:
        raise ValueError("Cannot use both upsample and downsample in res block")
    block_in, block_out = in_channels, out_channels
    block_stack = nn.Sequential()
    for _ in range(num_blocks):
        block_stack.append(
            res_block(block_in, block_out, dropout, norm_groups, norm_eps)
        )
        block_in = block_out
    if needs_downsample:
        block_stack.append(Downsample2D(out_channels))
    if needs_upsample:
        block_stack.append(Upsample2D(out_channels))
    return block_stack


def res_block(
    in_channels: int,
    out_channels: int,
    dropout: float = 0.0,
    norm_groups: int = 32,
    norm_eps: float = 1e-6,
) -> ResBlock:
    res_block_partial = partial(
        ResBlock,
        in_channels=in_channels,
        out_channels=out_channels,
        pre_outconv_dropout=dropout,
        scale_shift_conditional=False,
        norm_groups=norm_groups,
        norm_eps=norm_eps,
    )
    if in_channels != out_channels:
        return res_block_partial(
            skip_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    else:
        return res_block_partial()
