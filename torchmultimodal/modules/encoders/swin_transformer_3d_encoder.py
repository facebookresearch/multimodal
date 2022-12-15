# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Modified from 2d Swin Transformers in torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py


from typing import Callable, List, Optional

from torch import nn
from torchvision.models.video.swin_transformer import (
    PatchMerging,
    SwinTransformer3d as TVSwinTransformer3d,
)


class SwinTransformer3d(TVSwinTransformer3d):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    We upstream the model from torchvision:
    https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py#L363
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.0.
        num_classes (int, optional): Number of classes for classification head,
            if None it will have no head. Default: 400.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: Optional[int] = 400,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ):
        # Create non-optional _num_classes to construct torchvision SwinTransformer3d
        _num_classes = 400
        if num_classes is not None:
            _num_classes = num_classes

        super().__init__(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=_num_classes,
            norm_layer=norm_layer,
            block=block,
            downsample_layer=downsample_layer,
            patch_embed=patch_embed,
        )

        if num_classes is None:
            # Convert the head into identity
            self.head = nn.Identity()
