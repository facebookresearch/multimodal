# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from torch import nn, Tensor


class MDETRTextEmbeddings(nn.Module):
    """
    Class for RoBERTa embeddings as used in MDETR. This is very similar to the FLAVA
    text embeddings class except for its handling of position IDs.

    Args:   hidden_size (int): Embedding dimension. Default: 768
            vocab_size (int): Number of tokens in the vocabulary. Default: 30522
            pad_token_id (int): Index of padded tokens. Default: 0
            type_vocab_size (int): Number of token types. Default: 2
            max_position_embeddings (int): Max number of positions. Default: 512
            layer_norm_eps (float): Regularization value in layer norm. Default: 1e-12
            hidden_dropout_prob (float): Dropout probability on final embeddings.
                Default: 0.1

    Inputs: input_ids (Tensor): Tensor of input IDs to calculate embeddings for.
            token_type_ids (Optional[Tensor]): Optional tensor of token type IDs to use
                in token type embedding. Default: None
            position_ids (Optional[Tensor]): Optional tensor of position IDs to use in
                position embedding. Default: None
    """

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

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        position_ids_range = torch.arange(max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", position_ids_range.clone())
        self.register_buffer(
            "token_type_ids",
            torch.zeros(position_ids_range.size(), dtype=torch.long),
            persistent=False,
        )

    def create_position_ids_from_input_ids(
        self, input_ids: Tensor, padding_idx: int
    ) -> Tensor:
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Inputs: input_ids (Tensor): Tensor from which to create position IDs.
                padding_idx (int): Padding index
                    (determines starting point of position IDs).
        """
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + padding_idx

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        if position_ids is None:
            # Create the position ids from the input token ids. Any padded tokens remain padded.
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx
            )

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
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ModifiedTransformerEncoder(nn.Module):
    """
    Modified version of TorchText's RoBERTa transformer encoder
    taking in embeddings instead of input IDs.

    Args:   embedding_dim (int): Number of features in the input.
            num_encoder_layers  (int): Number of layers in the encoder.
            num_attention_heads (int): Number of heads in multi-head attention.
            ffn_dimension (int): Dimension of feedforward network inside
                attention layers.
            dropout (float): dropout value in each layer. Default: 0.1.
            normalize_before (bool): Whether to do PreNorm in encoder layers.
                Default: False
            return_all_layers (bool) Whether to return all layers (or just the last
                one). Default: False

    Inputs: embeddings (Tensor): Tensor of embeddings of a batch of input IDs.
            attention_mask (Optional[Tensor]) Batch attention mask returned from
                tokenizer (applied as padding mask inside self-attention).
                Default: None
    """ ""

    def __init__(
        self,
        embedding_dim: int,
        num_encoder_layers: int,
        num_attention_heads: int,
        ffn_dimension: int,
        dropout: float = 0.1,
        normalize_before: bool = False,
    ):
        super().__init__()
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
        self.embedding_dim = embedding_dim

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        encoded = embeddings
        # Do this in a loop because otherwise it can cause OOM
        for layer in self.layers.layers:
            encoded = layer(
                encoded, src_key_padding_mask=attention_mask.to(dtype=torch.bool)
            )
        return encoded


class MDETRTextEncoder(nn.Module):
    """
    Text encoder for MDETR. Combines an embedding module with a transformer encoder.

    Args:   embeddings (nn.Module): Embedding module (input IDs -> embeddings).
            encoder (nn.Module): Transformer encoder module
                (embeddings -> encoder outputs).

    Inputs: input_ids (Tensor): Tensor of input IDs to encode.
            attention_mask (Optional[Tensor]): Attention mask for batch. Should equal True
                on masked tokens on False on non-masked tokens. Default: None (no masking)
            token_type_ids (Optional[Tensor]): Optional tensor of token type IDs to use
                in token type embedding. Default: None
            position_ids (Optional[Tensor]): Optional tensor of position IDs to use in
                embeddings. Default: None
    """ ""

    def __init__(self, embeddings: nn.Module, encoder: nn.Module):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        out = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )

        return out


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    Args:   input_feat_size (int): Dimension of input features.
            output_feat_size (int): Dimension of output features.
            dropout (float): Dropout probability for final features. Default: 0.1
            do_ln (bool): Whether to perform layer normalization after the linear layer.
    Inputs: encoder_features (Tensor): Features to be resized.
    """

    def __init__(
        self,
        input_feat_size: int,
        output_feat_size: int,
        dropout: float = 0.1,
        do_ln: bool = True,
    ):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12) if do_ln else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: Tensor) -> Tensor:
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def mdetr_roberta_text_encoder(
    embedding_dim: int = 768,
    vocab_size: int = 50265,
    pad_token_id: int = 1,
    type_vocab_size: int = 1,
    max_position_embeddings: int = 514,
    layer_norm_eps: float = 1e-05,
    embedding_dropout_prob: float = 0.1,
    ffn_dimension: int = 3072,
    num_attention_heads: int = 12,
    num_encoder_layers: int = 12,
    encoder_dropout_prob: float = 0.1,
    normalize_before: bool = False,
) -> MDETRTextEncoder:
    embeddings = MDETRTextEmbeddings(
        hidden_size=embedding_dim,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        hidden_dropout_prob=embedding_dropout_prob,
    )

    modified_transformer_encoder = ModifiedTransformerEncoder(
        embedding_dim=embedding_dim,
        ffn_dimension=ffn_dimension,
        num_attention_heads=num_attention_heads,
        num_encoder_layers=num_encoder_layers,
        dropout=encoder_dropout_prob,
        normalize_before=normalize_before,
    )

    text_encoder = MDETRTextEncoder(
        embeddings=embeddings, encoder=modified_transformer_encoder
    )
    return text_encoder
