# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.encoders.bert_text_encoder import BERTTextEncoder

from torchmultimodal.modules.layers.text_embedding import BERTTextEmbeddings
from torchmultimodal.modules.layers.transformer import TransformerOutput


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
    """

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
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        encoded = embeddings
        batch_size, seq_len = embeddings.size()[:2]
        mask = attention_mask.reshape(batch_size, seq_len)

        # Do this in a loop because otherwise it can cause OOM
        for layer in self.layers.layers:
            encoded = layer(encoded, src_key_padding_mask=mask)
        return TransformerOutput(last_hidden_state=encoded)


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
) -> BERTTextEncoder:
    embeddings = BERTTextEmbeddings(
        hidden_size=embedding_dim,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        dropout=embedding_dropout_prob,
        offset_pos_ids=True,
    )

    modified_transformer_encoder = ModifiedTransformerEncoder(
        embedding_dim=embedding_dim,
        ffn_dimension=ffn_dimension,
        num_attention_heads=num_attention_heads,
        num_encoder_layers=num_encoder_layers,
        dropout=encoder_dropout_prob,
        normalize_before=normalize_before,
    )

    text_encoder = BERTTextEncoder(
        embeddings=embeddings, encoder=modified_transformer_encoder
    )
    return text_encoder
