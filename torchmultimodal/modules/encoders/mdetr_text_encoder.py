# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.text_embedding import TextEmbeddings


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
    embeddings = TextEmbeddings(
        hidden_size=embedding_dim,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        type_vocab_size=type_vocab_size,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        dropout=embedding_dropout_prob,
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
