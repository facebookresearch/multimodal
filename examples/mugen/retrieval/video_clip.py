# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from examples.mugen.retrieval.s3d import S3D
from torch import nn
from torchmultimodal.architectures.clip import CLIPArchitecture
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
        trainable (bool): whether the encoder's weights should be trainable (not frozen).
            Defaults to True. Ignored if ``pretrained`` is False.
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
        trainable: bool = True,
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

        if pretrained and not trainable:
            # check ``pretrained``` because if model isn't pretrained, then it should be trainable
            for p in self.model.parameters():
                p.requires_grad = False

    def build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.padding_value).to(dtype=int)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = self.build_attention_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class VideoEncoder(nn.Module):
    """Encode videos to a fixed size vector. Adapted from VideoCLIP
        (https://github.com/facebookresearch/fairseq/blob/main/examples/MMPT/mmpt/processors/models/s3dg.py)

    Args:
        trainable (bool): whether the encoder's weights should be trainable (not frozen).
            Defaults to True.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)

    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.model = S3D(400)
        self.model.fc = nn.Identity()

        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)


class Projection(nn.Module):
    """Project embeddings to a fixed dimension by adding the hidden-layer output and final output of a MLP.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
            Defaults to 256, the value used by MUGEN.
        dropout_prob (float): dropout probability.
            Defaults to 0.1, the value used by MUGEN.

    Inputs:
        x (Tensor): embeddings (batch, dim_in)

    Returns:
        Tensor: projected embeddings (batch, dim_out)

    """

    def __init__(self, in_dim, out_dim=256, dropout_prob=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.gelu(embed1)
        embed2 = self.linear2(embed2)
        embed2 = self.drop(embed2)
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


def videoclip(
    text_pretrained: bool = True,
    text_trainable: bool = True,
    text_model_name: str = "distilbert-base-uncased",
    text_model_config: Optional[Dict[str, Any]] = None,
    text_padding_value: int = 0,
    video_trainable: bool = True,
    proj_out_dim: int = 256,
    proj_dropout: float = 0.1,
) -> CLIPArchitecture:
    """Create MUGEN's video-text CLIP model with a S3D-backed video encoder and DistilBERT-backed text encoder.
        MUGEN paper: https://arxiv.org/abs/2204.08058

    Args:
        text_pretrained (bool): whether to use a pretrained text encoder or not.
            Defaults to True.
        text_trainable (bool): whether the text encoder's weights should be trainable.
            Defaults to True.
        text_model_name (str): name of pretrained model, used when pretrained is True.
            Defaults to "distilbert-base-uncased", Hugging Face's standard DistilBERT model.
        text_model_config (Optional[Dict[str, Any]]): model config for DistilBERT, used when pretrained is False
            Defaults to None, indicating the default DistilBERT config.
        text_padding_value (int): value that was used to pad the input text.
            Defaults to 0, Hugging Face's BERT pad token.
        video_trainable (bool): whether the video encoder's weights should be trainable.
            Defaults to True.
        proj_out_dim (int): output dimension to project both encoders' outputs to.
            Defaults to 256, the value used by MUGEN.
        proj_dropout (float): dropout probability in the projection layers.
            Defaults to 0.1, the value used by MUGEN.

    Returns:
        CLIPArchitecture: CLIP model using MUGEN's video encoder and text encoder.

    """
    text_model = TextEncoder(
        pretrained=text_pretrained,
        trainable=text_trainable,
        model_name=text_model_name,
        model_config=text_model_config,
        padding_value=text_padding_value,
    )
    text_encoder = nn.Sequential(
        text_model,
        Projection(
            text_model.model.config.dim, out_dim=proj_out_dim, dropout_prob=proj_dropout
        ),
    )
    video_encoder = nn.Sequential(
        VideoEncoder(
            trainable=video_trainable,
        ),
        Projection(1024, out_dim=proj_out_dim, dropout_prob=proj_dropout),
    )
    return CLIPArchitecture(encoder_a=text_encoder, encoder_b=video_encoder)
