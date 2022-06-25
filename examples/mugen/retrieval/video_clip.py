# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel


class TextEncoder(nn.Module):
    """Encode tokenized text to a fixed-length vector

    Adapted from MUGEN's text encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        pretrained (bool): whether to use a pretrained model or not
        model_name (str): name of pretrained model, used when pretrained is True
        model_config (dict): model config for DistilBERT, used when pretrained is False
        padding_value (int): value that was used to pad the input text

    Inputs:
        input_ids (Tensor): tensor of (batch, text_length) tokenized text

    Returns:
        Tensor: encoded text

    """

    def __init__(
        self,
        pretrained=True,
        model_name="distilbert-base-uncased",
        model_config=None,
        padding_value=0,
    ):
        super().__init__()
        self.padding_value = padding_value
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            distilbert_config = (
                DistilBertConfig(**model_config) if model_config else DistilBertConfig()
            )
            self.model = DistilBertModel(config=distilbert_config)

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.padding_value).to(dtype=int)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = self.build_attention_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
