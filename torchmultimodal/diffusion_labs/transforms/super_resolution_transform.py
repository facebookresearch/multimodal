# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

import torch.nn.functional as F

from torch import nn


class SuperResolutionTransform(nn.Module):
    """Data transform to generate small image for training with super resolution.

    Attributes:
        size (int): expected size of data
        low_res_size (int): size of scaled down data
        data_field (str): key name for data
        low_res_field (str): key name for low resolution data
        min_val (Optional[float]): min value for data
        max_val (Optional[float]): max value for data
        antilias (bool): whether to apply anti-aliasing when downsampling.
        mode (str): Interpolation mode to resizing
        align_corners (bool): align corners based on pixel pixel order instead of center.

    Args:
        x (Dict): data containing tensor "x".

    """

    def __init__(
        self,
        size: int,
        low_res_size: int,
        data_field: str = "x",
        low_res_field: str = "low_res",
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        # resize params
        antialias: bool = True,
        mode: str = "bicubic",
        align_corners: bool = False,
        augmentation_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.data_field = data_field
        self.low_res_field = low_res_field
        self.low_res_size = low_res_size
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.antialias = antialias
        self.mode = mode
        self.align_corners = align_corners
        self.augmentation_func = augmentation_func

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        assert (
            self.data_field in x
        ), f"{type(self).__name__} expects key {self.data_field}"
        data = x[self.data_field]
        # downsample
        down_scaled = F.interpolate(
            data,
            size=self.low_res_size,
            mode=self.mode,
            antialias=self.antialias,
            align_corners=self.align_corners,
        )
        if self.min_val or self.max_val:
            down_scaled = down_scaled.clamp(min=self.min_val, max=self.max_val)
        # augmentation
        if self.augmentation_func:
            down_scaled = self.augmentation_func(down_scaled)
        # upsample
        up_scaled = F.interpolate(
            down_scaled,
            size=self.size,
            mode=self.mode,
            antialias=self.antialias,
            align_corners=self.align_corners,
        )
        if self.min_val or self.max_val:
            up_scaled = up_scaled.clamp(min=self.min_val, max=self.max_val)
        x[self.low_res_field] = up_scaled
        return x
