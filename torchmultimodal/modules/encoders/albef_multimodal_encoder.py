# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn, Tensor
from torchmultimodal.modules.encoders.albef_text_encoder import (
    ALBEFTransformerAttention,
)
from torchmultimodal.utils.common import get_extended_attention_mask


class ALBEFMultimodalEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
        layer_norm_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                ALBEFTransformerLayerWithCrossAttention(
                    hidden_size,
                    intermediate_size,
                    num_attention_heads,
                    layer_norm_eps,
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
        if text_atts.size() == text_embeds.size()[:-1]:
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
    ) -> None:
        super().__init__()
        self.attention = ALBEFTransformerAttention(
            hidden_size, num_attention_heads, layer_norm_eps
        )
        self.cross_attention = ALBEFTransformerAttention(
            hidden_size, num_attention_heads, layer_norm_eps
        )
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.transform_act_fn = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
    ) -> Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.cross_attention(
            attention_output, attention_mask, encoder_hidden_states
        )
        dense1_output = self.dense1(attention_output)
        act_output = self.transform_act_fn(dense1_output)
        dense2_output = self.dense2(act_output)
        norm_output = self.layer_norm(dense2_output + attention_output)
        return norm_output
