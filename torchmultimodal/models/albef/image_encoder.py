# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch import nn
from torchmultimodal.modules.encoders.image_encoder import VisionTransformer
from torchmultimodal.modules.layers.image_embedding import ImageEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerEncoder


def albef_image_encoder(
    # TransformerEncoder params
    num_layers: int = 12,
    hidden_dim: int = 768,
    num_heads: int = 12,
    mlp_dim: int = 3072,
    activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-6,
    dropout: float = 0.0,
    # ImageEmbeddings params
    image_size: int = 256,
    patch_size: int = 16,
    num_channels: int = 3,
    use_image_masking: bool = False,
) -> VisionTransformer:
    embeddings = ImageEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_dim,
        hidden_dropout_prob=dropout,
        use_image_masking=use_image_masking,
    )
    encoder = TransformerEncoder(
        n_layer=num_layers,
        d_model=hidden_dim,
        n_head=num_heads,
        dim_feedforward=mlp_dim,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        norm_first=True,
    )

    layernorm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

    return VisionTransformer(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
    )
