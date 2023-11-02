# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm


class ResBlock(nn.Module):
    """Residual block in the ADM net. Supports projecting a conditional embedding to add to the hidden state.
    This typically contains the timestep embedding, but can also contain class embedding for classifier free guidance,
    CLIP image embedding and text encoder output for text-to-image generation as in DALL-E 2, or anything you want to
    condition the diffusion model on. If conditional embedding is not passed, the hidden state is simply passed through.

    Follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233) and BigGAN residual blocks (https://arxiv.org/abs/1809.11096).

    Code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L143


    Attributes:
        in_channels (int): num channels expected in input. Needs to be divisible by norm_groups.
        out_channels (int): num channels desired in output. Needs to be divisible by norm_groups.
        use_upsample (bool): include nn.Upsample layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_downsample is True.
        use_downsample (bool): include nn.AvgPool2d layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_upsample is True.
        activation (nn.Module): activation used before convs. Defaults to nn.SiLU().
        skip_conv (nn.Module): module used for additional convolution on skip connection. Defaults to nn.Identity().
        cond_proj (Optional[nn.Module]): module used for conditional embedding projection. Defaults to None.
        rescale_skip_connection (bool): whether to rescale skip connection by 1/sqrt(2), as described in "Diffusion
            Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233). Defaults to False.
        scale_shift_conditional (bool): if True, splits conditional embedding into two separate projections,
            and adds to hidden state as Norm(h)(w + 1) + b, as described in Appendix A in
            "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
            Defaults to True.
        pre_outconv_dropout (float): dropout probability before the second conv. Defaults to 0.1.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): Epsilon used in the GroupNorm layer. Defaults to 1e-5.


    Args:
        x (Tensor): input Tensor of shape [B x C x H x W]
        conditional_embedding (Tensor, optional): conditioning embedding vector of shape [B x C].
            If None, hidden state is passed through.

    Raises:
        TypeError: When skip_conv is not defined and in_channels != out_channels.
        TypeError: When use_upsample and use_downsample are both True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_upsample: bool = False,
        use_downsample: bool = False,
        activation: nn.Module = nn.SiLU(),
        skip_conv: nn.Module = nn.Identity(),
        cond_proj: Optional[nn.Module] = None,
        rescale_skip_connection: bool = False,
        scale_shift_conditional: bool = True,
        pre_outconv_dropout: float = 0.1,
        norm_groups: int = 32,
        norm_eps: float = 1e-05,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        if isinstance(skip_conv, nn.Identity) and in_channels != out_channels:
            raise ValueError(
                "You must specify a skip connection conv if out_channels != in_channels"
            )

        if in_channels % norm_groups != 0 or out_channels % norm_groups != 0:
            raise ValueError("Channel dims need to be divisible by norm_groups")

        if use_downsample and use_upsample:
            raise ValueError("Cannot use both upsample and downsample in res block")
        else:
            hidden_updownsample_layer: Union[nn.AvgPool2d, nn.Upsample, nn.Identity]
            skip_updownsample_layer: Union[nn.AvgPool2d, nn.Upsample, nn.Identity]
            if use_downsample:
                hidden_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
                skip_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            if use_upsample:
                hidden_updownsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
                skip_updownsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
            else:
                hidden_updownsample_layer = nn.Identity()
                skip_updownsample_layer = nn.Identity()

        self.cond_proj = cond_proj
        self.in_block = nn.Sequential(
            Fp32GroupNorm(
                norm_groups, in_channels, eps=norm_eps
            ),  # groups = 32 from code ref
            activation,
            hidden_updownsample_layer,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_group_norm = Fp32GroupNorm(norm_groups, out_channels, eps=norm_eps)
        self.out_block = nn.Sequential(
            activation,
            nn.Dropout(pre_outconv_dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip_block = nn.Sequential(
            skip_updownsample_layer,
            skip_conv,
        )

        self.scale_shift_conditional = scale_shift_conditional
        self.rescale_skip_connection = rescale_skip_connection

    def forward(
        self,
        x: Tensor,
        conditional_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        skip = self.skip_block(x)
        h = self.in_block(x)

        # Add conditional embedding to h, if they are passed and cond_proj is defined
        if conditional_embedding is not None and self.cond_proj is not None:
            t = self.cond_proj(conditional_embedding)
            # [b, c] -> [b, c, 1, 1]
            t = t.unsqueeze(-1).unsqueeze(-1)

            # If specified, split conditional embedding into two separate projections.
            # Use half to multiply with hidden state and half to add.
            # This is typically done after normalization.
            if self.scale_shift_conditional:
                h = self.out_group_norm(h)
                scale, shift = torch.chunk(t, 2, dim=1)
                h = h * (1 + scale) + shift
                h = self.out_block(h)
            else:
                h = self.out_block(self.out_group_norm(h + t))
        else:
            h = self.out_block(self.out_group_norm(h))

        if self.rescale_skip_connection:
            h = (skip + h) / 1.414
        else:
            h = skip + h
        return h


def adm_res_block(
    in_channels: int,
    out_channels: int,
    dim_cond: int,
    rescale_skip_connection: bool = False,
) -> ResBlock:
    if in_channels != out_channels:
        return adm_res_skipconv_block(in_channels, out_channels, dim_cond)
    return ResBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        rescale_skip_connection=rescale_skip_connection,
        cond_proj=adm_cond_proj(dim_cond, out_channels),
    )


def adm_res_downsample_block(
    num_channels: int,
    dim_cond: int,
    rescale_skip_connection: bool = False,
) -> ResBlock:
    return ResBlock(
        in_channels=num_channels,
        out_channels=num_channels,
        use_downsample=True,
        rescale_skip_connection=rescale_skip_connection,
        cond_proj=adm_cond_proj(dim_cond, num_channels),
    )


def adm_res_skipconv_block(
    in_channels: int,
    out_channels: int,
    dim_cond: int,
    rescale_skip_connection: bool = False,
) -> ResBlock:
    skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    return ResBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        skip_conv=skip_conv,
        rescale_skip_connection=rescale_skip_connection,
        cond_proj=adm_cond_proj(dim_cond, out_channels),
    )


def adm_res_upsample_block(
    num_channels: int,
    dim_cond: int,
    rescale_skip_connection: bool = False,
) -> ResBlock:
    return ResBlock(
        in_channels=num_channels,
        out_channels=num_channels,
        use_upsample=True,
        rescale_skip_connection=rescale_skip_connection,
        cond_proj=adm_cond_proj(dim_cond, num_channels),
    )


def adm_cond_proj(
    dim_cond: int,
    cond_channels: int,
    scale_shift_conditional: bool = True,
) -> nn.Module:
    if scale_shift_conditional:
        cond_channels *= 2
    return nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, cond_channels))
