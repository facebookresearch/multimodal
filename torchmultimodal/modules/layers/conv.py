# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from itertools import repeat
from typing import Any, Tuple, Union

from torch import nn, Size, Tensor
from torch.nn import functional as F


class SamePadConv3d(nn.Module):
    """Performs a same padded convolution on a 3D input.

    This maintains input shape with unit stride, and divides input dims by non-unit stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as ``nn.Conv3d``.
        out_channels (int): Number of channels for output, same as ``nn.Conv3d``.
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as ``nn.Conv3d``.
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as ``nn.Conv3d``.
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.pad_input: Tuple = None
        self.kernel_size = kernel_size
        self.stride = stride

        if "padding" in kwargs:
            warnings.warn(
                "Padding was specified but will not be used in favor of same padding, \
                use Conv3d directly for custom padding"
            )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=bias,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        # Calculate padding needed based on input shape only once to reduce run time
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(
                self.kernel_size, self.stride, x.shape[2:]
            )
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    """Performs a same padded transposed convolution on a 3D input.

    This ensures output shape in input shape multiplied by stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as Conv3d
        out_channels (int): Number of channels for output, same as Conv3d
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as Conv3d
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as Conv3d
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.pad_input: Tuple = None
        self.kernel_size = kernel_size
        self.stride = stride

        if "padding" in kwargs:
            warnings.warn(
                "Padding was specified but will not be used in favor of same padding, \
                use ConvTranspose3d directly for custom padding"
            )

        self.convt = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias, **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        # Calculate padding needed based on input shape only once to reduce run time
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(
                self.kernel_size, self.stride, x.shape[2:]
            )
            self.convt.padding, self.convt.output_padding = calculate_transpose_padding(
                self.kernel_size, self.stride, x.shape[2:], self.pad_input[::-1]
            )
        return self.convt(F.pad(x, self.pad_input))


def calculate_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    input_shape: Union[Size, Tuple[int, ...]],
) -> Tuple:
    """Calculates padding amount on each dimension based on given kernel size and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are halved. If
    stride does not divide into input evenly, then output = ceil(input / stride), following
    the TensorFlow implementation explained here:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.

    Returns:
        A tuple of the padding amount in a tuple of tuples for each dimension.
    """

    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))

    if not (len(kernel_size) == len(stride) == len(input_shape)):
        raise ValueError("dims for kernel, stride, and input must match")

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
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]],
    input_shape: Union[Size, Tuple[int, ...]],
    input_pad: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[Tuple, Tuple]:
    """Calculates padding for transposed convolution based on input dims, kernel size, and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are doubled.

    The 'padding' argument in ConvTranspose effectively trims the output, and the 'output_padding'
    argument effectively expands the output. These two knobs are adjusted to meet desired output dim.

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.
        input_pad (int or Tuple[int, ...]): Amount of padding added to input, must be twice length of
            kernel/stride/input_shape.

    Returns:
        A tuple of padding and output_padding to be used in ConvTranspose layers
    """

    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))
    if isinstance(input_pad, int):
        input_pad = tuple(repeat(input_pad, n_dims * 2))

    if not (len(kernel_size) == len(stride) == len(input_shape)):
        raise ValueError("dims for kernel, stride, and input must match")
    if len(input_pad) % 2 != 0 or len(input_pad) // 2 != len(input_shape):
        raise ValueError("input_pad length must be twice the number of dims")

    transpose_pad = []
    output_pad = []
    # Calculate current projected output dim and adjust padding and output_padding to match
    # input_dim * stride for a ConvTranspose layer
    for i, (d, k, s) in enumerate(zip(input_shape, kernel_size, stride)):
        # Calculate the output dim after transpose convolution:
        # out_dim = kernel + (in_dim + pad - 1) * stride
        # This needs to be adjusted with padding to meet desired dim, in_dim * stride
        output_shape_actual = k + (d + input_pad[2 * i] + input_pad[2 * i + 1] - 1) * s
        output_shape_expected = d * s
        # This controls padding argument in ConvTranspose,
        # where output dim is effectively trimmed by 2 * transpose_pad
        transpose_pad.append(
            max((output_shape_actual - output_shape_expected + 1) // 2, 0)
        )
        # This controls output_padding argument in ConvTranspose,
        # where output dim is expanded by 1 * output_pad
        output_pad.append(
            output_shape_expected - (output_shape_actual - transpose_pad[-1] * 2)
        )

    transpose_pad = tuple(transpose_pad)
    output_pad = tuple(output_pad)

    return transpose_pad, output_pad
