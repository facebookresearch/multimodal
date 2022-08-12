# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.transformer import TransformerOutput


class FLAVATransformerWithoutEmbeddings(nn.Module):
    # TODO(asg): Add support for pretrained checkpoint loading
    def __init__(
        self,
        encoder: nn.Module,
        layernorm: nn.Module,
        pooler: nn.Module,
        hidden_size: int = 768,
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
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
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
    ) -> TransformerOutput:
        if hidden_states is None:
            raise ValueError("You have to specify hidden_states")

        if self.cls_token is not None:
            batch_size = hidden_states.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        encoder_output = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            return_hidden_states=True,
            return_attn_weights=True,
        )
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return TransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
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
