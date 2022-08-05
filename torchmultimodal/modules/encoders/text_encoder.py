# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.text_embedding import TextEmbeddings
from torchmultimodal.modules.layers.transformer import (
    transformer_encoder,
    TransformerOutput,
)
from torchmultimodal.utils.attention import get_extended_attention_mask


class TextEncoder(nn.Module):
    """
    Construct word embeddings from input_ids and attention_mask.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Args:

    Inputs:
        input_ids (Tensor of size (batch_size, sequence_length)): Indices of input sequence tokens in the vocabulary.
        attention_mask (Tensor of shape (batch_size, sequence_length)): Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: Optional[nn.Module] = None,
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> TransformerOutput:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("input_ids or inputs_embeds must not be None")

        # only mask out padding token if no mask specified
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            attention_mask[input_ids == self.embeddings.pad_token_id] = 0

        # massage attention mask to correct shape for transformer
        attention_mask = get_extended_attention_mask(attention_mask)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask)

        sequence_output = encoder_output.last_hidden_state
        pooled_output = encoder_output.pooler_output
        if self.layernorm:
            sequence_output = self.layernorm(sequence_output)
        if self.pooler:
            pooled_output = self.pooler(sequence_output)

        return TransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


def text_encoder(
    # transformer encoder params
    hidden_size: int = 768,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    layer_norm_eps: float = 1e-12,
    norm_first: bool = False,
    # text embedding params
    vocab_size: int = 30522,
    max_position_embeddings: int = 512,
    type_vocab_size: int = 2,
    pad_token_id: int = 0,
) -> TextEncoder:
    embeddings = TextEmbeddings(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=type_vocab_size,
        layer_norm_eps=layer_norm_eps,
    )
    encoder = transformer_encoder(
        n_layer=num_hidden_layers,
        d_model=hidden_size,
        n_head=num_attention_heads,
        dim_feedforward=intermediate_size,
        activation=transform_act_fn,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
    )
    return TextEncoder(embeddings=embeddings, encoder=encoder)
