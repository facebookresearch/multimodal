# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, NamedTuple, Optional, Tuple

import torch

from torch import nn, Tensor
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.multi_head_attention import (
    MHAWithCacheOutput,
    MultiHeadAttentionWithCache,
    MultiHeadSelfAttention,
)
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm
from torchvision.ops.stochastic_depth import StochasticDepth


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[List[Tensor]] = None
    attentions: Optional[List[Tensor]] = None
    image_labels: Optional[Tensor] = None
    current_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer: transformer block consisting of multihead self-attention and feedforward blocks,
    based on "Attention Is All You Need" (Vaswani et al. 2017).

    Args:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention
            and feedforward. Otherwise, layer norm is done after. Defaults to False
        drop_path_rate (Optional[float]): use stochastic drop path instead of dropout for attn and feedforward dropout
        in transformer block as used by vision transformers https://arxiv.org/pdf/1603.09382.pdf. Defaults to None.
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
        drop_path_rate: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            embed_dim=d_model,
            num_heads=n_head,
        )
        if drop_path_rate is not None:
            self.attention_dropout = self.feedforward_dropout = StochasticDepth(
                drop_path_rate, mode="row"
            )
        else:
            self.attention_dropout = nn.Dropout(dropout)
            self.feedforward_dropout = nn.Dropout(dropout)

        self.feedforward = MLP(
            d_model, d_model, dim_feedforward, dropout=dropout, activation=activation
        )

        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.attention(
            hidden_states,
            attn_mask=attention_mask,
        )
        output = self.attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._attention_block(
            inputs,
            attention_mask=attention_mask,
        )
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(
            self.feedforward_layernorm(attn_residual)
        )

        return ff_residual

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = hidden_states
        attn_output = self._attention_block(
            x,
            attention_mask=attention_mask,
        )
        attn_residual = attn_output + x
        attn_residual = self.attention_layernorm(attn_residual)
        ff_residual = attn_residual + self._feedforward_block(attn_residual)
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            hidden_states (Tensor): input to the transformer encoder layer of shape bsz x seq_len x d_model
            attention_mask (Optional[Tensor]): attention mask of shape bsz x seq_len x seq_len.
            Same format as MultiHeadSelfAttention class.

        Returns:
            output tensor of shape bsz x seq_len x d_model
        """
        if self.norm_first is True:
            return self._forward_prenorm(
                hidden_states,
                attention_mask,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                attention_mask,
            )


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of n Transformer encoder layers and an optional final LN

    Args:
        n_layer (int): number of Transformer encoder layers
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention
            and feedforward. Otherwise, layer norm is done after. Defaults to False
        final_layer_norm_eps (Optional[float]): eps for final layer norm. Defaults to None.
        drop_path_rate (Optional[float]): use stochastic drop path instead of dropout for attn and feedforward dropout
        in transformer block sometimes used by vision transformers https://arxiv.org/pdf/1603.09382.pdf. Defaults to None.
    """

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
        drop_path_rate: Optional[float] = None,
    ):
        super().__init__()
        if drop_path_rate is not None:
            drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        else:
            drop_rate = [None for _ in range(n_layer)]
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
                    drop_rate[i],
                )
                for i in range(n_layer)
            ]
        )
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:

        """
        Args:
            hidden_states (Tensor): input to the transformer encoder of shape bsz x seq_len x d_model
            attention_mask (Optional[Tensor]): attention mask of shape bsz x seq_len x seq_len.
            Same format as MultiHeadSelfAttention class.
            return_hidden_states (bool): if True, return output from each layer of transformer including the input to first layer.
            Defaults to False.

        Returns:
            output of TransformerOutput type with the final output in last_hidden_state field.
            If return_hidden_states is set to True, the hidden_states field contains list of n_layer + 1 layer outputs.
            The last entry in the list is the output from last encoder block before final ln has been applied.
        """

        all_hidden_states = []

        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if return_hidden_states else None,
        )


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer consisting of multihead self-attention, optional
    cross-attention, and feedforward network.

    Args:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network.
            Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention,
            (optional) cross-attention, and feedforward. Otherwise, layer norm is done
            after. Defaults to False.
        use_cross_attention (bool): if True, cross-attention is applied before
            feedforward network. If False, no cross-attention is applied.
            Defaults to True.
        dim_kv (Optional[int]): dimension for key and value tensors in cross-attention.
            If None, K and V are assumed to have dimension d_model. Defaults to None.
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
        use_cross_attention: bool = True,
        dim_kv: Optional[int] = None,
    ) -> None:
        super().__init__()
        if dim_kv is not None:
            dim_kv = dim_kv
        else:
            dim_kv = d_model

        # Self-attention block
        self.attention = MultiHeadAttentionWithCache(
            dim_q=d_model,
            dim_kv=d_model,
            num_heads=n_head,
            dropout=dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)

        # Optional cross-attention block
        self.cross_attention: Optional[MultiHeadAttentionWithCache] = None
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attention = MultiHeadAttentionWithCache(
                dim_q=d_model,
                dim_kv=dim_kv,
                num_heads=n_head,
                dropout=dropout,
            )
            self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
            self.cross_attention_dropout = nn.Dropout(dropout)

        # Feedforward
        self.feedforward = MLP(
            d_model,
            d_model,
            dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.feedforward_dropout = nn.Dropout(dropout)

        # Layernorms
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        attn_output = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
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
        output = self.attention_dropout(attn_output_value)
        return output, present_key_value

    def _cross_attention_block(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert (
            self.cross_attention is not None
        ), """
            Cannot use cross-attention unless self.cross_attention and
            self.cross_attention_dropout are defined.
        """
        output = self.cross_attention(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            attn_mask=cross_attention_mask,
            # TODO: figure out caching for cross-attention
            use_cache=False,
        )
        assert torch.jit.isinstance(
            output, Tensor
        ), "cross-attention output must be Tensor."
        attention_output = self.cross_attention_dropout(output)
        return attention_output

    def _feedforward_block(self, hidden_states: Tensor) -> Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        # Self-attention
        self_attn_input = self.attention_layernorm(hidden_states)
        attn_output, present_key_value = self._self_attention_block(
            self_attn_input,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        self_attn_output = attn_output + hidden_states

        # Optional cross-attention
        if self.use_cross_attention:
            assert (
                encoder_hidden_states is not None
            ), "encoder_hidden_states must be provided for cross attention"
            assert hasattr(
                self, "cross_attention_layernorm"
            ), "Cross-attention layernorm not initialized"
            cross_attn_input = self.cross_attention_layernorm(self_attn_output)
            cross_attn_output = self._cross_attention_block(
                cross_attn_input,
                encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
            )
            attn_output = cross_attn_output + self_attn_output
        else:
            attn_output = self_attn_output

        # Feedforward
        ff_input = self.feedforward_layernorm(attn_output)
        output = attn_output + self._feedforward_block(ff_input)

        return output, present_key_value

    def _forward_postnorm(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        # Self-attention
        attn_output, present_key_value = self._self_attention_block(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        attn_residual = attn_output + hidden_states
        self_attn_output = self.attention_layernorm(attn_residual)

        # Optional cross-attention
        if self.use_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError(
                    "encoder_hidden_states must be provided for cross attention"
                )
            assert hasattr(
                self, "cross_attention_layernorm"
            ), "Cross-attention layernorm not initialized"
            cross_attn_output = self._cross_attention_block(
                self_attn_output, encoder_hidden_states, cross_attention_mask
            )
            cross_attn_residual = cross_attn_output + self_attn_output
            attn_output = self.cross_attention_layernorm(cross_attn_residual)
        else:
            attn_output = self_attn_output

        # Feedforward
        ff_residual = attn_output + self._feedforward_block(attn_output)
        output = self.feedforward_layernorm(ff_residual)

        return output, present_key_value

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape
                bsz x seq_len x embed_dim, only used for cross-attention.
                Default is None.
            attention_mask (Optional[Tensor]): attention mask for self-attention,
                supported mask type is described in MultiHeadAttentionWithCache class.
                Default is None.
            cross_attention_mask (Optional[Tensor]): attention mask for cross-attention,
                similar to attention_mask. Default is None.
            past_key_value (Optional[Tuple[Tensor, Tensor]]): cached key/value tuple
                for self-attention. Default is None.
            use_cache (bool): whether to use cache for key and value tensors.
                Can be used for faster autoregressive decoding during inference.
                    Default is False.

        Returns:
            A tuple including
                output (Tensor): layer output of shape bsz x seq_len x embed_dim
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple for
                    self-attention if use_cache set to True else None
        """
        if self.norm_first is True:
            return self._forward_prenorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
                past_key_value,
                use_cache,
            )
        else:
            return self._forward_postnorm(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
                past_key_value,
                use_cache,
            )


class TransformerDecoder(nn.Module):
    """
    Transformer decoder: n transformer decoder layers and an optional final LN

    Args:
        n_layer (int): number of transformer decoder layers
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network.
            Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention,
            (optional) cross-attention, and feedforward. Otherwise, layer norm is done
            after. Defaults to False.
        use_cross_attention (bool): if True, cross-attention is applied before
            feedforward network. If False, no cross-attention is applied.
            Defaults to True.
        dim_kv (Optional[int]): dimension for key and value tensors in cross-attention.
            If None, K and V are assumed to have dimension d_model. Defaults to None.
        final_layer_norm_eps (Optional[float]): epsilon used in final layer norm.
            Defaults to None (no final layer norm).
    """

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
        use_cross_attention: bool = True,
        dim_kv: Optional[int] = None,
        final_layer_norm_eps: Optional[float] = None,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    n_head,
                    dim_feedforward,
                    dropout,
                    activation,
                    layer_norm_eps,
                    norm_first,
                    use_cross_attention,
                    dim_kv,
                )
                for i in range(n_layer)
            ]
        )
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerOutput:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape
                bsz x seq_len x embed_dim, only used for cross-attention.
                Default is None.
            attention_mask (Optional[Tensor]): attention mask for self-attention,
                supported mask type is described in MultiHeadAttentionWithCache class.
                Default is None.
            cross_attention_mask (Optional[Tensor]): attention mask for cross-attention,
                similar to attention_mask. Default is None.
            past_key_values (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value
                tuples for self-attention in each layer. Default is None.
            use_cache (bool): whether to use cache for key and value tensors.
                Default is False.
            return_hidden_states (bool): if True, return output from each layer of
                transformer including the input to first layer. Default is False.

        Returns:
            output of TransformerOutput type with fields
                last_hidden_state (Tensor): layer output of shape bsz x seq_len x embed_dim
                hidden_states (List[Tensor]): all hidden states from decoder layers
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple
                    for self-attention.
        """

        all_hidden_states = []

        current_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])

        for i, layer_module in enumerate(self.layer):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs, current_key_value = layer_module(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                assert isinstance(current_key_value, tuple)
                current_key_values.append(current_key_value)

            hidden_states = layer_outputs

        if return_hidden_states:
            all_hidden_states.append(hidden_states)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            current_key_values=current_key_values,
        )
