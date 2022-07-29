# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable

from torch import nn, Tensor
from torchmultimodal.modules.layers.attention import MultiHeadAttention
from torchmultimodal.utils.attention import get_extended_attention_mask


class ALBEFMultimodalEncoder(nn.Module):
    """
    Construct multimodal embeddings from image embeddings, text embeddings, and text attention mask.

    The ALBEFMultimodalEncoder is similar to ALBEFTextEncoder, with the addition of image-text cross attention in encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers. Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder. Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder. Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        layer_norm_eps (float): The epsilon used by the layer normalization layers. Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer. Default is GELU.

    Inputs:
        image_embeds (Tensor of size (batch_size, image_seq_length, hidden_size)): Image embeddings from a vision encoder.
        text_embeds (Tensor of size (batch_size, text_seq_length, hidden_size)): Text embeddings from a text encoder.
        text_atts (Tensor of size (batch_size, text_seq_length)): Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                ALBEFTransformerLayerWithCrossAttention(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    layer_norm_eps,
                    transform_act_fn,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        text_atts: Tensor,
    ) -> Tensor:
        text_atts = get_extended_attention_mask(text_atts)
        hidden_states = text_embeds
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
            )
        return hidden_states


class ALBEFTransformerLayerWithCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        layer_norm_eps: float,
        transform_act_fn: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        # the attention block computes the self attention on text embeddings
        self.attention = MultiHeadAttention(
            dim_q=hidden_size,
            dim_kv=hidden_size,
            n_head=num_attention_heads,
        )
        # the cross_attention block computes the cross attention on image and text embeddings
        self.cross_attention = MultiHeadAttention(
            dim_q=hidden_size,
            dim_kv=hidden_size,
            n_head=num_attention_heads,
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.transform_act_fn = transform_act_fn
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        attention_output, _ = self.attention(
            hidden_states, attention_mask=attention_mask
        )
        attention_norm_output = self.attention_layer_norm(
            attention_output + hidden_states
        )
        cross_attention_output, _ = self.cross_attention(
            attention_norm_output,
            kv=encoder_hidden_states,
        )
        cross_attention_norm_output = self.cross_attention_layer_norm(
            cross_attention_output + attention_norm_output
        )
        dense1_output = self.dense1(cross_attention_norm_output)
        act_output = self.transform_act_fn(dense1_output)
        dense2_output = self.dense2(act_output)
        norm_output = self.ffn_layer_norm(dense2_output + cross_attention_norm_output)
        return norm_output
