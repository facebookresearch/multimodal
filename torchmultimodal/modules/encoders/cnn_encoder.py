# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """A CNN encoder.

    Stacks n layers of (Conv2d, MaxPool2d, BatchNorm2d), where n is determined
    by the length of the input args.

    Args:
        input_dims (List[int]): List of input dimensions.
        output_dims (List[int]): List of output dimensions. Should match
            input_dims offset by one.
        kernel_sizes (List[int]): Kernel sizes for convolutions. Should match
            the sizes of cnn_input_dims and cnn_output_dims.

    Inputs:
        x (Tensor): Tensor containing a batch of images.
    ​
    """

    def __init__(
        self, input_dims: List[int], output_dims: List[int], kernel_sizes: List[int]
    ):
        super().__init__()
        conv_layers: List[nn.Module] = []
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(
            kernel_sizes
        ), "input_dims, output_dims, and kernel_sizes should all have the same length"
        for in_channels, out_channels, kernel_size in zip(
            input_dims,
            output_dims,
            kernel_sizes,
        ):
            padding_size = kernel_size // 2

            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding_size
            )

            max_pool2d = nn.MaxPool2d(2, stride=2)
            batch_norm_2d = nn.BatchNorm2d(out_channels)

            conv_layers.append(
                nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d)
            )

        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)
