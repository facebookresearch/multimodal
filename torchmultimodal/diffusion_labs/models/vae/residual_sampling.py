# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

from torch import nn as nn, Tensor
from torch.nn import functional as F


class Upsample2D(nn.Module):
    """2-Dimensional upsampling layer with nearest neighbor interpolation and
    2D convolution, used for image decoders.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L91

    Attributes:
        channels (int): Number of channels in the input.

    Args:
        x (Tensor): 2-D image input tensor with shape (n, c, h, w).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Downsample2D(nn.Module):
    """2-Dimensional downsampling layer with zero padding and 2D convolution,
    used for image encoders.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L134

    Attributes:
        channels (int): Number of channels in the input.
        asymmetric_padding (bool): Whether to use asymmetric padding.
            Defaults to True.

    Args:
        x (Tensor): 2-D image input tensor with shape (n, c, h, w).
    """

    def __init__(
        self,
        channels: int,
        asymmetric_padding: bool = True,
    ):
        super().__init__()
        padding: Union[int, Tuple[int, int, int, int]]
        if asymmetric_padding:
            padding = (0, 1, 0, 1)
        else:
            padding = 1
        self.op = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)
