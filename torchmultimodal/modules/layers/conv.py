# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Tuple, Union

from torch import nn
from torch.nn import functional as F

from torchmultimodal.utils.common import calculate_same_padding


class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        self.pad_input = calculate_same_padding(kernel_size, stride)

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        self.pad_input = calculate_same_padding(kernel_size, stride)

        input_padding = tuple([k - 1 for k in kernel_size])
        input_padding = cast(Tuple[int, int, int], input_padding)
        output_padding = tuple(
            [4 - k if s == 2 else 0 for k, s in zip(kernel_size, stride)]
        )
        output_padding = cast(Tuple[int, int, int], output_padding)

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
            padding=input_padding,
            output_padding=output_padding,
        )  # only works for kernel=3 (besides default kernel=4) when stride=2

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))
