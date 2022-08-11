# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.transformer import TransformerEncoderLayer

FLAVATransformerOutput = namedtuple(
    "FLAVATransformerOutput",
    [
        "last_hidden_state",
        "pooler_output",
        "hidden_states",
        "attentions",
        "image_labels",
    ],
    defaults=(None, None, None, None, None),
)


class FLAVATransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        dropout: float = 0.0,
        intermediate_size: int = 3072,
        intermediate_activation: Callable[..., nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-12,
        **kwargs: Any,
    ):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    dropout=dropout,
                    activation=intermediate_activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_first=True,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> FLAVATransformerOutput:
        all_hidden_states = []
        all_self_attentions = []

        for i, layer_module in enumerate(self.layer):
            all_hidden_states.append(hidden_states)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                return_attn_weights=True,
            )

            hidden_states = layer_outputs[0]

            all_self_attentions.append(layer_outputs[1])

        all_hidden_states.append(hidden_states)

        all_hidden_states = tuple(all_hidden_states)
        all_self_attentions = tuple(all_self_attentions)

        return FLAVATransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class FLAVATransformerWithoutEmbeddings(nn.Module):
    # TODO(asg): Add support for pretrained checkpoint loading
    def __init__(
        self,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
        weight_init_fn: Optional[Callable] = None,
        initializer_range: float = 0.02,
        use_cls_token: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if use_cls_token:
            assert hasattr(
                encoder, "hidden_size"
            ), "hidden size not defined for given encoder"
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder.hidden_size))  # type: ignore
        else:
            self.cls_token = None

        if weight_init_fn is None:
            weight_init_fn = partial(
                init_transformer_weights, initializer_range=initializer_range
            )

        self.apply(weight_init_fn)

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> FLAVATransformerOutput:
        if hidden_states is None:
            raise ValueError("You have to specify hidden_states")

        if self.cls_token is not None:
            batch_size = hidden_states.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return FLAVATransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def init_transformer_weights(module: nn.Module, initializer_range: float) -> None:
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
