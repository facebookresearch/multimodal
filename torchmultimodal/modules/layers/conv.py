# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Tuple, Union

from torch import nn, Size, Tensor
from torch.nn import functional as F


class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape)
        return self.conv(F.pad(x, pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

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
        # TODO: refactor this to work for all kernel sizes and strides

    def forward(self, x) -> Tensor:
        pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape)
        return self.convt(F.pad(x, pad_input))


def calculate_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    input_shape: Union[Size, Tuple],
) -> Tuple[int, ...]:
    """Calculates padding amount on each dimension based on given kernel size and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are halved, and
    stride > 2 the output is ceil(input_dim // stride). Follows same padding notes from TF:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        kernel_size (int or Tuple): size of convolutional kernel
        stride (int or Tuple): stride amount of kernel

    Returns:
        Tuple: the padding amount in a tuple of tuples for each dimension
    """

    n_dims = len(input_shape[2:])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * n_dims
    if isinstance(stride, int):
        stride = (stride,) * n_dims

    total_pad = tuple(
        [max(k - (d % s), 0) for k, s, d in zip(kernel_size, stride, input_shape)]
    )
    pad_input = []
    for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
        pad_input.append((p // 2 + p % 2, p // 2))
    pad_input = tuple(sum(pad_input, tuple()))
    return pad_input
