# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Optional

import torch
from torch import nn
from torchmultimodal.architectures.omnivore import OmnivoreArchitecture
from torchmultimodal.modules.encoders.swin_transformer_3d_encoder import (
    PatchEmbed3d,
    SwinTransformer3d,
)

_OMNIVORE_PRETRAINED_URLS = {
    "swin_t": "https://download.pytorch.org/models/omnivore_swin_t-5b532aca.pth",
    "swin_s": "https://download.pytorch.org/models/omnivore_swin_s-b64cc260.pth",
    "swin_b": "https://download.pytorch.org/models/omnivore_swin_b-c2a4d126.pth",
}


def _imagenet1k_head(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1000, bias=True)


def _kinetics400_head(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(input_dim, 400, bias=True),
    )


def _sunrgbd_head(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 19, bias=True)


def _multimodal_head(input_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "image": _imagenet1k_head(input_dim),
            "rgbd": _sunrgbd_head(input_dim),
            "video": _kinetics400_head(input_dim),
        }
    )


class PatchEmbedOmnivore(nn.Module):
    """Patch Embedding strategy for Omnivore model
    It will use common PatchEmbed3d for image and video,
    for single view depth image it will have separate embedding for the depth channel
    and add the embedding result with the RGB channel
    reference: https://arxiv.org/abs/2201.08377

    Args:
        patch_size (Tuple[int, int, int]): Patch token size. Default: (2, 4, 4)
        embed_dim (int): Number of linear projection output channels. Default: 96
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3d(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        self.depth_patch_embed = PatchEmbed3d(
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C D H W
        # Note: D here represent time
        assert x.ndim == 5
        has_depth = x.shape[1] == 4

        if has_depth:
            x_rgb = self.patch_embed(x[:, :3, ...])
            x_d = self.depth_patch_embed(x[:, 3:, ...])
            x = x_rgb + x_d
        else:
            x = self.patch_embed(x)
        return x


def _omnivore_swin_t_encoder() -> SwinTransformer3d:
    encoder = SwinTransformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.2,
        norm_layer=nn.LayerNorm,
        patch_embed=PatchEmbedOmnivore,
        num_classes=None,
    )
    return encoder


def _omnivore_swin_s_encoder() -> SwinTransformer3d:
    encoder = SwinTransformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.3,
        norm_layer=nn.LayerNorm,
        patch_embed=PatchEmbedOmnivore,
        num_classes=None,
    )
    return encoder


def _omnivore_swin_b_encoder() -> SwinTransformer3d:
    encoder = SwinTransformer3d(
        patch_size=[2, 4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[16, 7, 7],
        stochastic_depth_prob=0.3,
        norm_layer=nn.LayerNorm,
        patch_embed=PatchEmbedOmnivore,
        num_classes=None,
    )
    return encoder


def omnivore_swin_t(
    encoder_only: bool = False, pretrained: bool = False, progress: bool = True
) -> nn.Module:
    """
    Builder function to get omnivore model with swin_t variant encoder
    Args:
        encoder_only (bool): If true then the builder will only return encoder without head (default: False)
        pretrained (bool): If true then the it will load pretrained weight,
            otherwise it will have random weight (default: False)
        progress (bool): If true then there will be a progress bar for downloading weight (default: True)
    """
    encoder = _omnivore_swin_t_encoder()
    heads = _multimodal_head(input_dim=encoder.num_features)
    model = OmnivoreArchitecture(encoder, heads)
    if pretrained:
        model.load_model(_OMNIVORE_PRETRAINED_URLS["swin_t"])
    if encoder_only:
        return model.encoder
    else:
        return model


def omnivore_swin_s(
    encoder_only: bool = False, pretrained: bool = False, progress: bool = True
) -> nn.Module:
    """
    Builder function to get omnivore model with swin_s variant encoder
    Args:
        encoder_only (bool): If true then the builder will only return encoder without head (default: False)
        pretrained (bool): If true then the it will load pretrained weight,
            otherwise it will have random weight (default: False)
        progress (bool): If true then there will be a progress bar for downloading weight (default: True)
    """
    encoder = _omnivore_swin_s_encoder()
    heads = _multimodal_head(input_dim=encoder.num_features)
    model = OmnivoreArchitecture(encoder, heads)
    if pretrained:
        model.load_model(_OMNIVORE_PRETRAINED_URLS["swin_s"])
    if encoder_only:
        return model.encoder
    else:
        return model


def omnivore_swin_b(
    encoder_only: bool = False, pretrained: bool = False, progress: bool = True
) -> nn.Module:
    """
    Builder function to get omnivore model with swin_b variant encoder
    Args:
        encoder_only (bool): If true then the builder will only return encoder without head (default: False)
        pretrained (bool): If true then the it will load pretrained weight,
            otherwise it will have random weight (default: False)
        progress (bool): If true then there will be a progress bar for downloading weight (default: True)
    """
    encoder = _omnivore_swin_b_encoder()
    heads = _multimodal_head(input_dim=encoder.num_features)
    model = OmnivoreArchitecture(encoder, heads)
    if pretrained:
        model.load_model(_OMNIVORE_PRETRAINED_URLS["swin_b"])
    if encoder_only:
        return model.encoder
    else:
        return model
