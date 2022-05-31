# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat
from typing import Tuple, Union

from torch import nn, Size, Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.modules.utils import _triple


class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.pad_input: Tuple = None
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(
                self.kernel_size, self.stride, x.shape[2:]
            )
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.pad_input: Tuple = None
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)

        self.convt = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x) -> Tensor:
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(
                self.kernel_size, self.stride, x.shape[2:]
            )
            self.convt.padding, self.convt.output_padding = calculate_transpose_padding(
                self.kernel_size, self.stride, x.shape[2:], self.pad_input
            )
        return self.conv(F.pad(x, self.pad_input))


def calculate_same_padding(
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    input_shape: Union[Size, Tuple],
) -> Tuple:
    """Calculates padding amount on each dimension based on given kernel size and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are halved. If
    stride does not divide into input evenly, then output = ceil(input / stride), following
    the TensorFlow implementation explained here:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        kernel_size (int or Tuple): size of convolutional kernel
        stride (int or Tuple): stride amount of kernel
        input_shape (Tuple or Size): tuple describing shape of input, without batch or channel dimension

    Returns:
        Tuple: the padding amount in a tuple of tuples for each dimension
    """

    assert (
        len(kernel_size) == len(stride) == len(input_shape)
    ), "dims for kernel, stride, and input must match"

    total_pad = []
    for k, s, d in zip(kernel_size, stride, input_shape):
        if d % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - (d % s), 0)
        total_pad.append(pad)
    pad_input = []
    for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
        pad_input.append(p // 2 + p % 2)
        pad_input.append(p // 2)
    pad_input = tuple(pad_input)
    return pad_input


def calculate_transpose_padding(
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    input_shape: Union[Size, Tuple],
    input_pad: Tuple[int, ...] = None,
) -> Tuple[Tuple, Tuple]:
    """Calculates

    Args:
        kernel_size (Tuple): size of convolutional kernel
        stride (Tuple): stride amount of kernel
        input_shape (Tuple or Size): tuple describing shape of input, without batch or channel dimension
        input_pad (Tuple): amount of padding added to input, must be twice length of kernel/stride/input_shape

    Returns:
        Tuple: padding and output_padding to be used in ConvTranspose layers
    """

    assert (
        len(kernel_size) == len(stride) == len(input_shape)
    ), "dims for kernel, stride, and input must match"
    if input_pad is None:
        input_pad = tuple(repeat(0, len(input_shape) * 2))
    assert len(input_pad) % 2 == 0 and len(input_pad) // 2 == len(
        input_shape
    ), "input_pad length must be twice the number of dims"

    transpose_pad = []
    output_pad = []
    # Calculate current projected output dim and adjust padding and output_padding to match
    # input_dim * stride for a ConvTranspose layer
    for i, (d, k, s) in enumerate(zip(input_shape, kernel_size, stride)):
        output_shape_actual = k + (d + input_pad[2 * i] + input_pad[2 * i + 1] - 1) * s
        output_shape_expected = d * s
        transpose_pad.append(
            max((output_shape_actual - output_shape_expected + 1) // 2, 0)
        )
        output_pad.append(
            output_shape_expected - (output_shape_actual - transpose_pad[-1] * 2)
        )

    transpose_pad = tuple(transpose_pad)
    output_pad = tuple(output_pad)

    return transpose_pad, output_pad
