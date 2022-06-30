# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from examples.mugen.retrieval.s3d import S3D
from torch import nn

from torchmultimodal.utils.common import PretrainedMixin
from transformers import DistilBertConfig, DistilBertModel


PRETRAINED_S3D_KINETICS400_URL = (
    "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/S3D_kinetics400.pt"
)


class TextEncoder(nn.Module):
    """Encode tokenized text to the last hidden state representation of the CLS token using
        DistilBERT. DistilBERT prepends a CLS (classification) token to every text so the
        token's hidden state represents the entire text.

    Adapted from MUGEN's text encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        pretrained (bool): whether to use a pretrained model or not.
            Defaults to True.
        model_name (str): name of pretrained model, used when pretrained is True.
            Defaults to "distilbert-base-uncased", Hugging Face's standard DistilBERT model.
        model_config (Optional[Dict[str, Any]]): model config for DistilBERT, used when pretrained is False
            Defaults to None, indicating the default DistilBERT config.
        padding_value (int): value that was used to pad the input text.
            Defaults to 0, Hugging Face's BERT pad token.

    Inputs:
        input_ids (Tensor): tensor of (batch, text_length) tokenized text

    Returns:
        Tensor: encoded text with dimensions (batch, model_config.dim).
            Default and pretrained model_config.dim is 768.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_name: str = "distilbert-base-uncased",
        model_config: Optional[Dict[str, Any]] = None,
        padding_value: int = 0,
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


class VideoEncoder(nn.Module, PretrainedMixin):
    """Encode videos to a fixed size vector.

    Adapted from MUGEN's video encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        pretrained (bool): whether to use a pretrained model or not.
            Defaults to True.
        pretrain_path (str): local path or remote URL to pretrained weights.
            Defaults to ``PRETRAINED_S3D_KINETICS400_URL``, the weights MUGEN used from
            pretraining S3D on Kinetics 400. Ignored if ``pretrained`` is False.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(
        self,
        pretrained: bool = True,
        pretrain_path: str = PRETRAINED_S3D_KINETICS400_URL,
    ):
        super().__init__()
        self.model = S3D(400)
        self.model.fc = nn.Identity()

        if pretrained:
            self.load_model(pretrain_path)

    def forward(self, x):
        return self.model(x)
