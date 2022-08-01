# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from torch import nn, Tensor
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.vision_transformer import VisionTransformer


class ALBEFVisionEncoder(nn.Module):
    """
    Modified VisionTransformer used by ALBEF.

    This class returns the output of the encoder ('encoder.ln'), without passing it to the heads.

    Args:
        image_size (int): The size (resolution) of each image.
            Default is 256.
        patch_size (int) The size (resolution) of each patch.
            Default is 16.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 12.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
            Default is 768.
        mlp_dim (int): Dimensionality of the MLP Block in the encoder layers.
            Default is 3072.
        dropout (float): The dropout ratio for the encoder probabilities.
            Default is 0.
        attention_dropout (float): The dropout ratio for the attention probabilities.
            Default is 0.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-6.

    Inputs:
        x (Tensor): Tensor of size (n, c, image_size, image_size) containing image features
    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        vision_transformer = VisionTransformer(
            image_size,
            patch_size,
            num_hidden_layers,
            num_attention_heads,
            hidden_size,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps),
        )
        self.encoder_layer_name = "encoder.ln"
        self.encoder = create_feature_extractor(
            vision_transformer, [self.encoder_layer_name]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)[self.encoder_layer_name]
