# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchmultimodal.diffusion_labs.modules.adapters.adapter import Adapter
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


class SuperResolution(nn.Module, Adapter):
    """
    A wrapper for super resolution training. Takes a low resolution input
    from the conditional input and scales it accordingly to be appended to the input.
    The new input will have 2x total channels, and the inner model should expect this larger input.

    Attributes:
        model (nn.Module): the neural network
        low_res_name (str): the key for the small image in conditional_inputs
        antialias (bool): whether to apply antialiasing to the output
        mode (str): the pytorch downsampling mode used for small images
        align_corners (bool): whether to align corners
        min_val (Optional[float]): clamp values below this
        max_val (Optional[float]): clamp values above this

    Args:
        x (Tensor): low res image input Tensor of shape [b, in_channels, ...]
        timestep (Tensor): diffusion step
        conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
            Conditional embeddings must contain key `small_img`.
    """

    def __init__(
        self,
        model: nn.Module,
        low_res_name: str = "low_res",
        # torch.nn.functional.interpolate params
        antialias: bool = True,
        mode: str = "bicubic",
        align_corners: bool = False,
        # clamp params
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.model = model
        self.low_res_name = low_res_name
        self.antialias = antialias
        self.mode = mode
        self.align_corners = align_corners
        self.min_val = min_val
        self.max_val = max_val

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> DiffusionOutput:
        if conditional_inputs:
            low_res = conditional_inputs.get(self.low_res_name, torch.zeros_like(x))
        else:
            low_res = torch.zeros_like(x)
        # upsample to target resolution if not already at target resolution
        if x.size() != low_res.size():
            low_res = F.interpolate(
                low_res,
                size=tuple(x.size()[2:]),
                mode=self.mode,
                antialias=self.antialias,
                align_corners=self.align_corners,
            )
            if self.min_val or self.max_val:
                low_res = low_res.clamp(min=self.min_val, max=self.max_val)
        x = torch.cat([x, low_res], dim=1)
        return self.model(x, timestep)
