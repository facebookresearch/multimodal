# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm


class AttentionResBlock(nn.Module):
    """Attention block in the LDM Autoencoder that consists of group norm, attention,
    conv projection and a residual connection.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#LL150C1-L150C6

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of
            q, k, v in attention module. Needs to be divisible by norm_groups.
        attn_module (nn.Module): Module of attention mechanism to use.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `num_channels` is not divisible by `norm_groups`.
    """

    def __init__(
        self,
        num_channels: int,
        attn_module: nn.Module,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        if num_channels % norm_groups != 0:
            raise ValueError("Channel dims need to be divisible by norm_groups")

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("norm", Fp32GroupNorm(norm_groups, num_channels, norm_eps)),
                    ("attn", attn_module),
                    ("out", nn.Conv2d(num_channels, num_channels, kernel_size=1)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x


class VanillaAttention(nn.Module):
    """Attention module used in the LDM Autoencoder. Similar to standard Q, k V attention,
    but using 2d convolutions instead of linear projections for obtaining q, k, v tensors.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#LL150C1-L150C6

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of q, k, v.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
    """

    def __init__(self, num_channels: int):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.query = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.key = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.value = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        B, C, H, W = q.shape
        # [B, C, H, W] -> [B, H*W, C]
        q, k, v = (t.reshape(B, C, H * W).permute(0, 2, 1) for t in (q, k, v))
        # [B, H*W, C]
        out = F.scaled_dot_product_attention(q, k, v)
        # [B, H*W, C] -> [B, C, H, W]
        return out.permute(0, 2, 1).reshape(B, C, H, W)


def attention_res_block(
    channels: int,
    norm_groups: int = 32,
    norm_eps: float = 1e-6,
) -> AttentionResBlock:
    return AttentionResBlock(
        channels, VanillaAttention(channels), norm_groups, norm_eps
    )
