# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable

from torch import nn, Tensor
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.vision_transformer import VisionTransformer


class ALBEFVisionEncoder(nn.Module):
    """
    Modified VisionTransformer used by ALBEF.

    Based on https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L160.
    This class removes the heads from VisionTransformer.

    Args:
        image_size (int): The size (resolution) of each image
        patch_size (int) The size (resolution) of each patch
        num_layers (int): Number of hidden layers in the Transformer encoder
        num_heads (int): Number of attention heads for each attention layer in the Transformer encoder
        hidden_dim (int): Dimensionality of the encoder layers and the pooler layer
        mlp_dim (int): Dimensionality of the MLP Block in the encoder layers
        dropout (float): The dropout ratio for the encoder probabilities
        attention_dropout (float): The dropout ratio for the attention probabilities
        norm_layer (Callable[..., torch.nn.Module]): The normalization layer in the encoder layers

    Inputs:
        x (Tensor): Tensor of size (n, c, image_size, image_size) containing image features
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        vision_transformer = VisionTransformer(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer=norm_layer,
        )
        self.encoder = create_feature_extractor(vision_transformer, ["encoder.ln"])

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)["encoder.ln"]
