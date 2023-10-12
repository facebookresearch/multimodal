# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

from torch import nn, Tensor
from torchmultimodal.models.blip2.qformer_layers import QformerEmbedding, QformerEncoder

from torchmultimodal.models.blip2.qformer_utils import get_causal_mask


class QformerModel(nn.Module):
    """
    Qformer model including Qformer embedding and Qformer encoder.

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        query_length(int): query length in Qformer, used to compute cached query length.
            default value is the same as num_query_token for Blip2 case (https://fburl.com/316803mo).
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA, default is None.
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        dim_q: int,
        dim_feedforward: int,
        num_heads: int,
        max_position_embeddings: int,
        vocab_size: int,
        pad_token_id: int = 0,
        query_length: int = 32,
        dim_kv: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
        activation: Callable[..., nn.Module] = nn.ReLU,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        cross_attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.query_length = query_length
        self.embeddings = QformerEmbedding(
            embedding_dim=dim_q,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        self.encoder = QformerEncoder(
            num_hidden_layers=num_hidden_layers,
            dim_q=dim_q,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            cross_attention_freq=cross_attention_freq,
            dim_kv=dim_kv,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        query_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        use_causal_mask: bool = False,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids for QFormer
            attention_mask (Optional[Tensor]): attention mask for QFormer
            position_ids (Optional[Tensor]): position ids for QFormer
            query_embeds (Optional[Tensor]): query embeddings for QFormer
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            past_key_values: (Optional[List[Tuple[Tensor, Tensor]]]):  a list of num_layers elements,
                each element is a 2-element tuple for cached key/value.
                key/value is tensor with shape of (bsz x source_seq_len x embed_dim).
            use_cache (bool): whether to use cache for key and value tensors
            use_causal_mask (bool): apply causal mask if true, default to False

        Returns:
            Qformer encoder output with a tuple of last hidden states and past_key_values if use_cache.
        """
        past_seq_length = (
            # overall_seq_length - query_length
            past_key_values[0][0].shape[2] - self.query_length
            if past_key_values is not None
            else 0
        )
        query_length = query_embeds.shape[1] if query_embeds is not None else 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            query_embeddings=query_embeds,
            past_seq_length=past_seq_length,
        )
        bsz, seq_len = embedding_output.size()[:-1]

        if attention_mask is not None:
            if use_causal_mask:
                # Apply a causal mask in addition to the padding mask and make attention mask broadcastable.
                causal_mask = get_causal_mask(
                    attention_mask,
                    (bsz, seq_len),
                    has_query=(query_embeds is not None),
                )
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
                attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
            else:
                attention_mask = attention_mask[:, None, None, :]
            # create a tensor which is 0.0 for positions to attend and -10000.0 for masked position.
            # use float mask to ensure mask values will be added to the attention weight
            attention_mask = (1.0 - attention_mask) * -10000.0

        return self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            query_length=query_length,
        )


class QformerPredictionHead(nn.Module):
    """
    MLP head for computinng prediction score from QformerModel output

    Args:
        dim_q (int): dimensionality of the query tensor
        vocab_size (int): the size of vocabulary used by QFormer
        layer_norm_eps (float): the epsilon used by the layer normalization layers, default is 1e-12
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
    """

    def __init__(
        self,
        dim_q: int,
        vocab_size: int,
        layer_norm_eps: float = 1e-12,
        activation: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim_q, dim_q)
        self.activation = activation()
        self.layernorm = nn.LayerNorm(dim_q, eps=layer_norm_eps)
        self.linear_2 = nn.Linear(dim_q, vocab_size)

    def forward(self, sequence_output: Tensor) -> Tensor:
        """
        Inputs (Tensor):
            sequence_output of shape bsz x seq_len x embed_dim
        Returns:
            prediction scores (Tensor) of shape: bsz x seq_len x vocab_size
        """
        hidden_states = self.linear_1(sequence_output)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        predictions = self.linear_2(hidden_states)
        return predictions


class QformerForCLM(nn.Module):
    """
    A QformerModel wrapper class for causal language modeling(clm).

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        query_length(int): query length in Qformer, details see QformerModel class.
        dim_kv (Optional[int]): dim_kv (Optional[int]): dimensions of the key and value tensors, this value is only used in CA.
            Default is None.
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2
    """

    def __init__(
        self,
        num_hidden_layers: int,
        dim_q: int,
        dim_feedforward: int,
        num_heads: int,
        max_position_embeddings: int,
        vocab_size: int,
        pad_token_id: int = 0,
        query_length: int = 32,
        dim_kv: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
        activation: Callable[..., nn.Module] = nn.GELU,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        cross_attention_freq: int = 2,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.head = QformerPredictionHead(
            dim_q=dim_q,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            vocab_size=vocab_size,
        )
        self.model = QformerModel(
            num_hidden_layers=num_hidden_layers,
            dim_q=dim_q,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            query_length=query_length,
            dim_kv=dim_kv,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            attn_dropout=attn_dropout,
            dropout=dropout,
            cross_attention_freq=cross_attention_freq,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        query_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tensor:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids for QFormer
            attention_mask (Optional[Tensor]): attention mask for QFormer
            position_ids (Optional[Tensor]): position ids for QFormer
            query_embeds (Optional[Tensor]): query embeddings for QFormer
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            past_key_values: (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value tuple for self-attention
            use_cache (bool): whether to use cache for key and value tensors,
                default to False for generation as cached values should be computed in previous training tasks.

        Returns:
            prediction score (Tensor) computed for next word prediction of shape
                bsz x seq_len x vocab_size
        """
        # TODO: revisit if it's required for edge cases after BLIP-2 impl.
        if past_key_values is not None:
            assert query_embeds is None

        sequence_output, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            use_causal_mask=True,  # set causal mask for clm
        )
        if query_embeds is not None:
            sequence_output = sequence_output[:, query_embeds.shape[1] :, :]

        prediction_scores = self.head(sequence_output)
        return prediction_scores
