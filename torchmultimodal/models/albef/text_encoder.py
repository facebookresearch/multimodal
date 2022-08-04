# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import (
    transformer_encoder,
    TransformerOutput,
)
from torchmultimodal.utils.attention import get_extended_attention_mask


class ALBEFTextEncoder(nn.Module):
    """
    Construct word embeddings from input_ids and attention_mask.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Args:
        vocab_size (int): Vocabulary size of the model. Defines the different tokens that can be represented by the inputs_ids.
            Default is 30522.
        hidden_size (int): Dimensionality of the encoder layers. Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder. Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder. Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        max_position_embeddings (int): The maximum sequence length that this model might ever be used with. Default is 512.
        type_vocab_size (int): The vocabulary size of the token_type_ids. Default is 2.
        pad_token_id (int): The embedding for pad_token_id is not updated during training. Default is 0.
        layer_norm_eps (float): The epsilon used by the layer normalization layers. Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer. Default is GELU.
    Inputs:
        input_ids (Tensor of size (batch_size, sequence_length)): Indices of input sequence tokens in the vocabulary.
        attention_mask (Tensor of shape (batch_size, sequence_length)): Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.embeddings = ALBEFTextEmbeddings(
            vocab_size,
            hidden_size,
            pad_token_id,
            max_position_embeddings,
            type_vocab_size,
            layer_norm_eps,
        )
        self.encoder = transformer_encoder(
            n_layer=num_hidden_layers,
            d_model=hidden_size,
            n_head=num_attention_heads,
            dim_feedforward=intermediate_size,
            activation=transform_act_fn,
            layer_norm_eps=layer_norm_eps,
            norm_first=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> TransformerOutput:
        extended_attention_mask = get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)

        return encoder_outputs


class ALBEFTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, input_ids: Tensor) -> Tensor:
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        device = input_ids.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        return embeddings
