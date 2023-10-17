# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
from torch import nn, Tensor

from torchmultimodal.diffusion_labs.modules.adapters.adapter import Adapter
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


class InPainting(nn.Module, Adapter):
    """Inpainting masks a target artifact during generation to infill the masked region. This approach
    is based on "Palette: Image-to-Image Diffusion Models" (https://arxiv.org/abs/2111.05826)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with
    Text-Guided Diffusion Models" (https://arxiv.org/abs/2112.10741). A mask is input as a
    conditional input and concatenated with the input. The new input will have
    2 * channels + 1 total channels, and the inner model should expect this larger input.

    Attributes:
        model (nn.Module): the neural network
        mask_name (str): the key for the mask in conditional_inputs
        target_name (str): the key for the target artifcact in conditional_inputs

    Args:
        x (Tensor): input Tensor of shape [b, in_channels, ...]
        timestep (Tensor): diffusion step
        conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
            Conditional embeddings must have at least 2 dimensions.
    """

    def __init__(
        self, model: nn.Module, mask_name: str = "mask", target_name: str = "target"
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.model = model
        self.mask_name = mask_name
        self.target_name = target_name

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> DiffusionOutput:
        if (
            conditional_inputs
            and self.target_name in conditional_inputs
            and self.mask_name in conditional_inputs
        ):
            target = conditional_inputs[self.target_name]
            mask = conditional_inputs[self.mask_name]
            masked_target = target * mask
        else:
            mask = torch.zeros_like(x[:, :1])
            masked_target = torch.zeros_like(x)
        masked_x = torch.cat([x, masked_target, mask], dim=1)
        return self.model(masked_x, timestep, conditional_inputs)
