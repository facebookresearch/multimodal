# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.layers.normalizations import FrozenBatchNorm2d
from torchmultimodal.utils.common import NestedTensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet101, ResNet101_Weights


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# At this point this class is basically just a wrapper
# around the backbone to support tensor list and IntermediateLayerGetter.
# We can refactor to pass tensor and mask separately
class MDETRBackbone(nn.Module):
    def __init__(self, body: nn.Module, intermediate_layers: Dict[str, int]):
        super().__init__()
        # Note that we need this to skip pooler, flatten, and FC layers in
        # the standard ResNet implementation.
        self.body = IntermediateLayerGetter(body, return_layers=intermediate_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = []
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out.append(NestedTensor(x, mask))
        return out


def mdetr_resnet101_backbone() -> MDETRBackbone:
    # TODO: maybe support passing arg for last dilation operation as in MDETR repo
    body = resnet101(
        replace_stride_with_dilation=[False, False, False],
        weights=ResNet101_Weights.IMAGENET1K_V1,
        norm_layer=FrozenBatchNorm2d,
    )

    backbone = MDETRBackbone(body, intermediate_layers={"layer4": 0})
    return backbone
