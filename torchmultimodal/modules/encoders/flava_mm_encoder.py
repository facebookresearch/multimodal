# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch import nn, Tensor
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.layers.transformer import (
    FLAVATransformerWithoutEmbeddings,
    FLAVATransformerEncoder,
)
from torchmultimodal.modules.losses.flava import Pooler


def flava_multimodal_encoder(
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    num_hidden_layers: int = 12,
    hidden_dropout_prob: float = 0.0,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., Tensor] = nn.functional.gelu,
    attention_probs_dropout_prob: float = 0.0,
    layer_norm_eps: float = 1e-12,
):
    encoder = FLAVATransformerEncoder(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermediate_size,
        intermediate_activation=intermediate_activation,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)

    return FLAVATransformerWithoutEmbeddings(
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
    )
