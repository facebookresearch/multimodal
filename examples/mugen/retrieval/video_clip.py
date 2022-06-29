# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from typing import Any, Dict, Optional

import torch
from examples.mugen.retrieval.s3d import S3D
from torch import nn

from torchmultimodal.utils.common import get_current_device
from transformers import DistilBertConfig, DistilBertModel


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


class VideoEncoder(nn.Module):
    """Encode videos to a fixed size vector.

    Adapted from MUGEN's video encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        pretrained (bool): whether to use a pretrained model or not.
            Defaults to True.
        pretrain_file (Optional[Any]): a file-like object (has to implement read, readline, tell, and seek),
            or a string or os.PathLike object containing a file name.
            Defaults to None, which loads the weights MUGEN used from pretraining S3D on Kinetics 400.
            Ignored if ``pretrained`` is False.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(self, pretrained: bool = True, pretrain_file: Optional[Any] = None):
        super().__init__()
        self.model = S3D(400)
        self.model.fc = nn.Identity()

        if pretrained:
            weight_dict = self.get_pretrained_weights(pretrain_file)
            model_dict = self.model.state_dict()
            for name, param in weight_dict.items():
                if "fc.0" not in name:
                    if "module" in name:
                        name = ".".join(name.split(".")[1:])
                    model_dict[name].copy_(param)

    def get_model_dir(self, url):
        return os.path.join(
            torch.hub.get_dir(),
            "multimodal",
            hashlib.sha256(url.encode("utf-8")).hexdigest(),
        )

    def get_pretrained_weights(self, file: Optional[Any] = None) -> dict:
        """Loads a pretrained weights dict for the S3D encoder.

        Args:
            file (Optional[Any]): a file-like object (has to implement read, readline, tell, and seek),
                or a string or os.PathLike object containing a file name.
                Defaults to None, which loads the weights MUGEN used from pretraining S3D on Kinetics 400.

        Returns:
            dict: dictionary of weights for loading into the VideoEncoder

        """
        if not file:
            pretrained_url = "https://drive.google.com/uc?export=download&id=1HJVDBOQpnTMDVUM3SsXLy0HUkf_wryGO"
            weight_dict = torch.hub.load_state_dict_from_url(
                pretrained_url,
                model_dir=self.get_model_dir(pretrained_url),
                map_location=get_current_device(),
            )
        else:
            weight_dict = torch.load(file, map_location=get_current_device())
        return weight_dict

    def forward(self, x):
        return self.model(x)
