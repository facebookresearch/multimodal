# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

import torch

from torch import nn, Tensor

from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.multi_head_attention import (
    MHAWithCacheOutput,
    MultiHeadAttentionWithCache,
)
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm


class QformerLayer(nn.Module):
    """
    Qformer layer module.

    This module is designed with a self-attention (SA) block and optionally includes a cross-attention (CA) block for queries.
    The inputs for this module, referred to as hidden_states, can consist of either a query, text, or a combination of both.
    Cross-attention is exclusively activated for queries (query_length > 0) with encoder_hidden_states derived from image inputs.

    The feedforward(ff) block will project the hidden states output by the layer before,
    query output and text output are concatenated as overall output after separated handling for CA and ff.

    Args:
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        has_cross_attention (bool): whether a cross-attention layer is included
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA.

    """

    def __init__(
        self,
        dim_q: int,
        dim_feedforward: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
        activation: Callable[..., nn.Module] = nn.ReLU,
        has_cross_attention: bool = False,
        dim_kv: Optional[int] = None,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttentionWithCache(
            dim_q, dim_q, num_heads, attn_dropout
        )
        self.self_attn_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.has_cross_attention = has_cross_attention
        self.cross_attention: Optional[MultiHeadAttentionWithCache] = None

        if has_cross_attention:
            # If this is instantiated as a cross-attention module, the keys
            # and values come from an encoder, key and value caching should be disabled.
            if dim_kv is None:
                raise ValueError(
                    "key and value dim should be provided for cross attention."
                )
            self.cross_attention = MultiHeadAttentionWithCache(
                dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads, dropout=attn_dropout
            )
            self.cross_attn_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
            self.cross_attn_dropout = nn.Dropout(dropout)

        # feedforward block
        self.feedforward = MLP(
            dim_q, dim_q, dim_feedforward, dropout=0.0, activation=activation
        )
        self.feedforward_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.feedforward_dropout = nn.Dropout(dropout)

        # query feedforward block
        self.feedforward_query = MLP(
            dim_q, dim_q, dim_feedforward, dropout=0.0, activation=activation
        )
        self.feedforward_layernorm_query = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.feedforward_dropout_query = nn.Dropout(dropout)

    def _self_attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        x = hidden_states
        attn_output = self.self_attention(
            x,
            x,
            x,
            attn_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        present_key_value: Optional[Tuple[Tensor, Tensor]] = None
        if use_cache:
            assert isinstance(attn_output, MHAWithCacheOutput)
            attn_output_value = attn_output.attn_output
            present_key_value = attn_output.past_key_value
        else:
            assert isinstance(attn_output, Tensor)
            attn_output_value = attn_output
        attn_output = self.dropout(attn_output_value)

        attn_residual = attn_output + x
        attn_residual = self.self_attn_layernorm(attn_residual)
        return attn_residual, present_key_value

    def _cross_attention_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        x = hidden_states
        assert self.cross_attention is not None
        # turn off cache for cross attention
        cross_attn_output = self.cross_attention(
            query=x,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            use_cache=False,
        )

        if not torch.jit.isinstance(cross_attn_output, Tensor):
            raise ValueError("cross-attention output must be Tensor.")
        cross_attn_output = self.cross_attn_dropout(cross_attn_output)
        cross_attn_residual = cross_attn_output + x
        cross_attn_residual = self.cross_attn_layernorm(cross_attn_residual)
        return cross_attn_residual

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        h = self.feedforward_layernorm(h + hidden_states)
        return h

    def _feedforward_query_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward_query(hidden_states)
        h = self.feedforward_dropout_query(h)
        h = self.feedforward_layernorm_query(h + hidden_states)
        return h

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        query_length: int = 0,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            attention_mask (Optional[Tensor]): attention mask, supported mask type is described in MultiHeadAttentionWithCache class
            past_key_value (Optional[Tuple[Tensor, Tensor]]): cached key/value tuple for self-attention
            query_length (Optional[int]): length of query embedding, used as condition
                to determine query attention output and check text existance.
            use_cache (bool): whether to use cache for key and value tensors

        Return:
            A tuple includes:
                layer_output (Tensor): layer output of shape bsz x seq_len x embed_dim
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple for self-attention
        """
        if past_key_value is not None and len(past_key_value) != 2:
            raise ValueError(
                "past_key_value should be 2-element tuple to represent self-attention cached key/values."
            )
        attn_residual, present_key_value = self._self_attention_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        if query_length > 0:
            query_attn_output = attn_residual[:, :query_length, :]
            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError(
                        "encoder_hidden_states must be given for cross-attention layers"
                    )
                cross_attn_output = self._cross_attention_block(
                    hidden_states=query_attn_output,
                    encoder_hidden_states=encoder_hidden_states,
                )
                query_attn_output = cross_attn_output

            # add query feedforward block for self-attention or cross-attention
            layer_output = self._feedforward_query_block(query_attn_output)

            # handle text input if present
            if attn_residual.shape[1] > query_length:
                layer_output_text = self._feedforward_block(
                    attn_residual[:, query_length:, :]
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)

        else:
            layer_output = self._feedforward_block(attn_residual)

        return (layer_output, present_key_value)


class QformerEncoder(nn.Module):
    """
    Qformer encoder module including multiple Qformer layers.

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2.
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA.

    """

    def __init__(
        self,
        num_hidden_layers: int,
        dim_q: int,
        dim_feedforward: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-12,
        activation: Callable[..., nn.Module] = nn.ReLU,
        cross_attention_freq: int = 2,
        dim_kv: Optional[int] = None,
    ):
        super().__init__()
        layers = []
        for i in range(num_hidden_layers):
            has_cross_attention = i % cross_attention_freq == 0
            layers.append(
                QformerLayer(
                    dim_q=dim_q,
                    dim_feedforward=dim_feedforward,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    activation=activation,
                    has_cross_attention=has_cross_attention,
                    dim_kv=dim_kv,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        query_length: int = 0,
        use_cache: bool = False,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            attention_mask (Optional[Tensor]): attention mask, supported mask type is described in MultiHeadAttentionWithCache class
            past_key_values (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value tuple for self-attention
            query_length (int): the length of input query, used for cross-attention
            use_cache (bool): whether to use cache for key and value tensors

        Return:
            A tuple includes:
                the last hidden state: Tensor of shape bsz x seq_len x embed_dim
                past_key_values (List[Optional[Tuple[Tensor, Tensor]]]]): cached key/values from Qformer layers
        """
        current_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for i, layer_module in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, current_key_value = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_value,
                query_length=query_length,
                use_cache=use_cache,
            )
            if use_cache:
                assert isinstance(current_key_value, tuple)
                current_key_values.append(current_key_value)

        return (hidden_states, current_key_values)


class QformerEmbedding(nn.Module):
    """
    Qformer embedding module.

    Args:
        embedding_dim (int): dim of embedding space
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        dropout (float): dropout probability after embedding layers and layernorm.
        layer_norm_eps (float): the epsilon used by the layer normalization layers.
    """

    def __init__(
        self,
        embedding_dim: int,
        max_position_embeddings: int,
        vocab_size: int,
        pad_token_id: int = 0,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.layernorm = Fp32LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        query_embeddings: Optional[Tensor] = None,
        past_seq_length: int = 0,
    ) -> Tensor:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids
            position_ids (Optional[Tensor]): batches of of 1D integer tensors used to identify each token's position,
                if no position_ids is provided, the IDs are automatically created as absolute positional embeddings.
            query_embeddings (Optional[Tensor]): query embeddings for QFormer
            past_seq_length (Optional[int]): sequence length cached by past_key_values.

        Returns:
            embeddings (Tensor): concatenated embeddings of shape (bsz, num tokens, embedding dim), concatenation is along
            the token dimension.
        """
        if input_ids is None and query_embeddings is None:
            raise ValueError("Either input_ids or query_embeddings must be passed.")

        seq_length = input_ids.size(1) if input_ids is not None else 0

        embeddings = query_embeddings

        if input_ids is not None:
            if position_ids is None:
                position_ids = self.position_ids[
                    :, past_seq_length : seq_length + past_seq_length
                ].clone()
            word_embeddings = self.token_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids.long())
            embeddings = word_embeddings + position_embeddings

            if query_embeddings is not None:
                embeddings = torch.cat((query_embeddings, embeddings), dim=1)

        assert isinstance(embeddings, Tensor)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
