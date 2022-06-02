# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from torch import nn, Tensor


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


# TODO (later): Possibly refactor into a common class with FLAVA text embeddings class
class MDETRTextEmbeddings(nn.Module):
    """This class is identical to FLAVA's TextEmbeddings class except for its handling of position IDs."""

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        type_vocab_size: int = 2,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id
        )
        self.padding_idx = pad_token_id
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=self.padding_idx
        )
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        position_ids_range = torch.arange(max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", position_ids_range.clone())
        self.register_buffer(
            "token_type_ids",
            torch.zeros(position_ids_range.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = create_position_ids_from_input_ids(
                input_ids, self.padding_idx
            )

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]  # type: ignore
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    batch_size,
                    seq_length,
                    dtype=torch.long,
                    device=device,
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Wrap TorchText's TransformerEncoder to take in embeddings instead of tokens
class WrappedTransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_encoder_layers: int,
        num_attention_heads: int,
        ffn_dimension: Optional[int] = None,
        dropout: float = 0.1,
        normalize_before: bool = False,
        return_all_layers: bool = False,
    ):
        super().__init__()
        ffn_dimension = ffn_dimension or 4 * embedding_dim
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=ffn_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=normalize_before,
        )
        self.layers = torch.nn.TransformerEncoder(
            encoder_layer=layer, num_layers=num_encoder_layers
        )
        self.normalize_before = normalize_before
        self.embedding_layer_norm = nn.LayerNorm(embedding_dim)
        self.return_all_layers = return_all_layers

    def _forward_return_all_layers(
        self,
        embeddings: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        encoded = embeddings
        states = [encoded]
        for layer in self.layers.layers:
            encoded = layer(encoded, src_key_padding_mask=attn_mask)
            states.append(encoded)
        if self.normalize_before:
            for i, state in enumerate(states):
                states[i] = self.embedding_layer_norm(state)
        return states

    def _forward_return_last_layer(
        self,
        embeddings: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = embeddings
        for layer in self.layers.layers:
            encoded = layer(encoded, src_key_padding_mask=attn_mask)
        if self.normalize_before:
            encoded = self.embedding_layer_norm(encoded)
        return encoded

    def forward(
        self,
        embeddings: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        out: Union[torch.Tensor, List[torch.Tensor]]
        if self.return_all_layers:
            out = self._forward_return_all_layers(embeddings, attn_mask, padding_mask)
        else:
            out = self._forward_return_last_layer(embeddings, attn_mask, padding_mask)

        return out


class MDETRTextEncoder(nn.Module):
    def __init__(self, embeddings: nn.Module, encoder: nn.Module):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder

    # Note: MDETR's RoBERTa encoder also returns pooler outputs in forward, but
    # these are only used for contrastive loss. Since contrastive loss is not used anywhere,
    # we omit the pooler for the time being.
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        # Cast mask to bool and invert it
        # In PyTorch attention masks, True means the position is masked,
        # while in Hugging Face transformers the masks are inverted.
        attention_mask = ~attention_mask.bool()

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        out = self.encoder(
            embedding_output,
            attn_mask=attention_mask,
        )

        return out
