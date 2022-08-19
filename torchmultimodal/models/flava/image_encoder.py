# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.models.flava.transformer import init_transformer_weights
from torchmultimodal.modules.encoders.image_encoder import VisionTransformer
from torchmultimodal.modules.layers.image_embedding import ImageEmbeddings
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerOutput,
)
from torchmultimodal.modules.losses.flava import Pooler


def flava_image_encoder(
    # TransformerEncoder params
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    dropout: float = 0.0,
    # ImageEmbeddings params
    image_size: int = 224,
    patch_size: int = 16,
    num_channels: int = 3,
    use_image_masking: bool = False,
    # VisionTransformer params
    initializer_range: float = 0.02,
) -> VisionTransformer:

    embeddings = ImageEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=num_channels,
        hidden_size=hidden_size,
        hidden_dropout_prob=dropout,
        use_image_masking=use_image_masking,
    )
    encoder = TransformerEncoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        activation=intermediate_activation,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
        norm_first=True,
    )

    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    weight_init_fn = partial(
        init_transformer_weights, initializer_range=initializer_range
    )

    return VisionTransformer(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
        weight_init_fn=weight_init_fn,
    )


class ImageTransformerWithVAE(nn.Module):
    def __init__(
        self,
        image_transformer: nn.Module,
        vae: nn.Module,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.image_transformer = image_transformer
        self.vae = vae

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        image_patches_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> TransformerOutput:
        image_labels = self.vae(pixel_values).flatten(1)
        image_patches_mask = image_patches_mask.flatten(1).to(torch.bool)
        image_labels[image_patches_mask == False] = -1  # noqa

        output = self.image_transformer(
            pixel_values=pixel_values,
            image_patches_mask=image_patches_mask,
            attention_mask=attention_mask,
        )
        return TransformerOutput(
            last_hidden_state=output.last_hidden_state,
            pooler_output=output.pooler_output,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
