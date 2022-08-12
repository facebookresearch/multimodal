# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models._api import Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import resnet101, ResNet101_Weights


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copied from torchvision.ops.misc with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans. This module is a useful replacement for BatchNorm2d in the
    case of very small batches, see https://bit.ly/3xQvmiJ.


    Args:   n (int): Number of features ``C`` from expected input size ``(N, C, H, W)``
            eps (float): Value added to denominator for numerical stability.
                Default = 1e-5

    Inputs: x (Tensor): Tensor to be normalized
    """

    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning to make it fuser-friendly
        # ignore mypy errors because fixing them would break checkpoint loading
        w = self.weight.reshape(1, -1, 1, 1)  # type: ignore
        b = self.bias.reshape(1, -1, 1, 1)  # type: ignore
        rv = self.running_var.reshape(1, -1, 1, 1)  # type: ignore
        rm = self.running_mean.reshape(1, -1, 1, 1)  # type: ignore
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbedding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762),
    generalized to work on images.

    Args:   num_pos_feats (int): Number of positional features
                (should be half the output embedding size). Default = 64
            temperature (int): Base for generating frequency mesh. Default = 10000
            scale (float): Scaling factor when performing normalization. Setting
                scale = s will rescale values to fall in [0, s].
                Default = None (no normalization)

    Inputs: mask (Tensor): Padding mask (used to infer size of each image in batch).
                Input size: (batch_size, height, width)

    Returns: Tensor of size (batch_size, 2 * num_pos_feats, height, width)
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self, mask: Tensor) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.scale is not None:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=mask.device
        )
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


class MaskedIntermediateLayer(nn.Module):
    """
    This class wraps a backbone returning an intermediate layer (e.g. a ResNet
    where we do not want to perform pooling) while casting masks to the appropriate
    sizes.

    Note: for simplicity we only support returning a single intermediate layer.

    Args:   body (nn.Module): The module to return the intermediate layer from.
            intermediate_layer (str): Name of the layer to return from body.

    Inputs: images (Tensor): Batch of images to pass to the backbone.
            image_masks (Tensor): Masks to cast to backbone output size.
    """

    def __init__(self, body: nn.Module, intermediate_layer: str):
        super().__init__()
        # Note that we need this to skip pooler, flatten, and FC layers in
        # the standard ResNet implementation.
        self.body = IntermediateLayerGetter(body, return_layers={intermediate_layer: 0})

    def forward(
        self, images: torch.Tensor, image_masks: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        out = self.body(images)
        tensor = out[next(iter(out))]
        mask = F.interpolate(image_masks[None].float(), size=tensor.shape[-2:]).bool()[
            0
        ]
        return tensor, mask


def mdetr_resnet101_backbone(
    weights: Weights = ResNet101_Weights.IMAGENET1K_V1,
    norm_layer: Callable[..., nn.Module] = FrozenBatchNorm2d,
    freeze_weights: bool = True,
) -> MaskedIntermediateLayer:
    body = resnet101(
        replace_stride_with_dilation=[False, False, False],
        weights=weights,
        norm_layer=norm_layer,
    )
    if freeze_weights:
        for name, parameter in body.named_parameters():
            if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
    backbone = MaskedIntermediateLayer(body, intermediate_layer="layer4")
    return backbone
