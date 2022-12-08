# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Code for some of the transformers components in this file are initialized
# from their counterparts in Hugging Face Transformers library.

from typing import Callable, NamedTuple, Optional, Tuple, Union

from torch import nn, Tensor
from torchmultimodal.modules.layers.attention import MultiHeadAttention, SelfAttention
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attentions: Optional[Tuple[Tensor, ...]] = None
    image_labels: Optional[Tensor] = None


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


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer is made up of multihead self-attention and feedforward blocks,
    based on the architecture in "Attention Is All You Need" (Vaswani et al. 2017). Similar to
    ``nn.TransformerEncoderLayer``, but uses a custom ``MultiHeadAttention`` that supports
    n-dimensional inputs (including sequences, images, video) and head-masking.

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
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``. See
            ``MultiHeadAttention`` for shape requirements.
        head_mask (Tensor, optional): mask to be applied to self-attention inputs after softmax and dropout,
            before matrix multiplication with values. See ``MultiHeadAttention`` for shape requirements.
        return_attn_weights (bool, optional): return attention probabilities in addition to attention output.
            Defaults to False.
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
        # feedforward block
        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )
        self.feedforward_dropout = nn.Dropout(dropout)
        # layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output, attn_weights = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_attn_weights=True,
        )
        output = self.attention_dropout(output)
        return output, attn_weights

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output, attn_weights = self._attention_block(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(
            self.feedforward_layernorm(attn_residual)
        )
        if return_attn_weights:
            return ff_residual, attn_weights
        else:
            return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = hidden_states
        attn_output, attn_weights = self._attention_block(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        attn_residual = attn_output + x
        attn_residual = self.attention_layernorm(attn_residual)
        ff_residual = attn_residual + self._feedforward_block(attn_residual)
        outputs = self.feedforward_layernorm(ff_residual)
        if return_attn_weights:
            return outputs, attn_weights
        else:
            return outputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.norm_first:
            return self._forward_prenorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                attention_mask,
                head_mask,
                return_attn_weights,
            )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        layer_norm_eps: float = 1e-12,
        norm_first: bool = False,
        final_layer_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    n_head,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    norm_first,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        all_hidden_states: Tuple[Tensor, ...] = () if return_hidden_states else None
        all_self_attentions: Tuple[Tensor, ...] = () if return_attn_weights else None

        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                return_attn_weights=return_attn_weights,
            )

            if return_attn_weights:
                hidden_states = layer_outputs[0]
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

        if return_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
