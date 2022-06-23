# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchmultimodal.utils.common import get_current_device
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer


class TextEncoder(nn.Module):
    """Encode text to a fixed-length vector

    Taken from MUGEN's text encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)
    """

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        pretrained=True,
        max_length=200,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, raw_text):
        device = get_current_device()
        batch_encoding = self.tokenizer(
            raw_text, padding=True, truncation=True, max_length=self.max_length
        )
        input_ids = torch.tensor(batch_encoding["input_ids"]).to(device)
        attention_mask = torch.tensor(batch_encoding["attention_mask"]).to(device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
