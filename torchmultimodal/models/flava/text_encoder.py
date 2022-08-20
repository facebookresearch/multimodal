# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable

from torch import nn
from torchmultimodal.models.flava.transformer import init_transformer_weights
from torchmultimodal.modules.encoders.bert_text_encoder import BERTTextEncoder
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.modules.layers.text_embedding import BERTTextEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerEncoder
from torchmultimodal.modules.losses.flava import Pooler


def flava_text_encoder(
    # TransformerEncoder params
    num_hidden_layers: int = 12,
    hidden_size: int = 768,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    intermediate_activation: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    dropout: float = 0.0,
    # TextEmbeddings params
    vocab_size: int = 30522,
    pad_token_id: int = 0,
    type_vocab_size: int = 2,
    max_position_embeddings: int = 512,
    # TextEncoder params
    initializer_range: float = 0.02,
) -> BERTTextEncoder:

    embeddings = BERTTextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        dropout=dropout,
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
    return BERTTextEncoder(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
        pooler=pooler,
        weight_init_fn=weight_init_fn,
    )
