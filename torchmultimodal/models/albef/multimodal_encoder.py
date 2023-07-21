# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Optional

from torch import nn, Tensor
from torchmultimodal.modules.layers.attention import MultiHeadAttention, SelfAttention
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchmultimodal.utils.attention import get_extended_attention_mask


class TransformerCrossAttentionLayer(nn.Module):
    """Transformer layer with self-attention on inputs and cross-attention on an encoder's outputs.
    Can be used in a transformer decoder or an encoder with cross-attention. Similar to
    ``nn.TransformerDecoderLayer``, but generalized for use in an encoder with cross-attention as well.
    Uses a custom ``MultiHeadAttention`` that supports n-dimensional inputs including sequences,
    images, video.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        encoder_hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate
            cross-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``.
            See ``MultiHeadAttention`` for shape requirements.
        cross_attention_mask (Tensor, optional): mask to be applied to cross-attention inputs,
            ``encoder_hidden_states``. See ``MultiHeadAttention`` for shape requirements.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        # attention block
        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.attention_dropout = nn.Dropout(dropout)
        # cross attention block
        self.cross_attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=SelfAttention(dropout),
        )
        self.cross_attention_dropout = nn.Dropout(dropout)
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(
        self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        output = self.attention(
            hidden_states, attention_mask=attention_mask, return_attn_weights=False
        )
        output = self.attention_dropout(output)
        return output

    def _cross_attention_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            attention_mask=cross_attention_mask,
            return_attn_weights=False,
        )
        output = self.cross_attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._self_attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.cross_attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.feedforward_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        attn_output = self._self_attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(
            attn_norm_output, kv, cross_attention_mask
        )
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.cross_attention_layernorm(
            cross_attention_residual
        )
        ff_residual = cross_attention_norm_output + self._feedforward_block(
            cross_attention_norm_output
        )
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
            )


class ALBEFMultimodalEncoder(nn.Module):
    """
    Construct multimodal embeddings from image embeddings, text embeddings, and text attention mask.

    The ALBEFMultimodalEncoder is similar to ALBEFTextEncoder, with the addition of image-text cross attention in encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers.
            Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer.
            Default is GELU.

    Inputs:
        hidden_states (Tensor of shape (batch_size, seq_len, hidden_size)):
            Unimodal input hidden states.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Optional[Tensor] of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.
        is_decoder (bool): Whether this module is used as a decoder. Default is False.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    activation=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attention_mask = get_extended_attention_mask(attention_mask)
        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)

        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_mask=encoder_attention_mask,
            )
        return hidden_states
