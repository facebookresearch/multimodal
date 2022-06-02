# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from torchmultimodal.modules.layers.normalizations import FrozenBatchNorm2d
from torchmultimodal.utils.common import NestedTensor
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet101_Weights

# At this point this class is basically just a wrapper
# around the backbone to support tensor list and IntermediateLayerGetter.
# We should refactor to pass tensor and mask separately
class MDETRBackbone(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        # Note that we need this to skip pooler, flatten, and FC layers in
        # the standard ResNet implementation.
        self.body = IntermediateLayerGetter(body, return_layers={"layer4": 0})

    def forward(self, tensor_list):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


def _mdetr_resnet101_backbone():
    # TODO: maybe support passing arg for last dilation operation as in MDETR repo
    body = getattr(torchvision.models, "resnet101")(
        replace_stride_with_dilation=[False, False, False],
        weights=ResNet101_Weights.IMAGENET1K_V1,
        norm_layer=FrozenBatchNorm2d,
    )

    return MDETRBackbone(body)
