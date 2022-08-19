# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Dict, Optional

import torch
from examples.mugen.retrieval.s3d import S3D
from torch import nn

from torchmultimodal.models.clip.model import CLIP
from torchmultimodal.utils.common import load_module_from_url
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
        model_config (Optional[Dict[str, Any]]): model config for DistilBERT.
            Defaults to ``None``, indicating the default DistilBERT config.
        padding_value (int): value that was used to pad the input text.
            Defaults to ``0``, Hugging Face's BERT pad token.

    Inputs:
        input_ids (Tensor): tensor of (batch, text_length) tokenized text

    Returns:
        Tensor: encoded text with dimensions (batch, ``model_config.dim``).
            Default ``model_config.dim`` is ``768``.
    """

    def __init__(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        padding_value: int = 0,
    ):
        super().__init__()
        self.padding_value = padding_value
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        distilbert_config = (
            DistilBertConfig(**model_config) if model_config else DistilBertConfig()
        )
        self.model = DistilBertModel(config=distilbert_config)
        self.out_dim = self.model.config.dim

    def build_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.padding_value).to(dtype=int)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = self.build_attention_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class VideoEncoder(nn.Module):
    """Encode videos to the last layer before the fully-connected layer of S3D.

    Adapted from MUGEN's video encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
            Size of ``channel`` dimension must be ``3``.

    """

    def __init__(
        self,
    ):
        super().__init__()
        self.model = S3D(400)
        self.out_dim = self.model.fc.in_channels
        self.model.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] != 3:
            raise ValueError(
                "Channels must be at first (zero-indexed) dimension of input and of size 3."
            )
        return self.model(x)


class Projection(nn.Module):
    """Project embeddings to a fixed dimension by adding the hidden-layer output and final output of a MLP.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
            Defaults to ``256``, the value used by MUGEN.
        dropout_prob (float): dropout probability.
            Defaults to ``0.1``, the value used by MUGEN.

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
    video_pretrained: bool = True,
    video_trainable: bool = True,
    video_pretrain_path: str = PRETRAINED_S3D_KINETICS400_URL,
    proj_out_dim: int = 256,
    proj_dropout: float = 0.1,
) -> CLIP:
    """Create MUGEN's video-text CLIP model with a S3D-backed video encoder and DistilBERT-backed text encoder.
        MUGEN paper: https://arxiv.org/abs/2204.08058

    Args:
        text_pretrained (bool): whether to use a pretrained text encoder or not.
            Defaults to ``True``.
        text_trainable (bool): whether the text encoder's weights should be trainable.
            Defaults to ``True``. Ignored if ``text_pretrained`` is ``False``.
        text_model_name (str): name of pretrained model, used when ``text_pretrained`` is ``True``.
            Defaults to ``"distilbert-base-uncased"``, Hugging Face's standard DistilBERT model.
        text_model_config (Optional[Dict[str, Any]]): model config for DistilBERT.
            Defaults to ``None``, indicating the default DistilBERT config.
        text_padding_value (int): value that was used to pad the input text.
            Defaults to ``0``, Hugging Face's BERT pad token.
        video_pretrained (bool): whether to use a pretrained model or not.
            Defaults to ``True``.
        video_trainable (bool): whether the video encoder's weights should be trainable.
            Defaults to ``True``. Ignored if ``video_pretrained`` is ``False``.
        video_pretrain_path (str): local path or remote URL to video encoder pretrained weights.
            Defaults to ``PRETRAINED_S3D_KINETICS400_URL``, the weights MUGEN used from
            pretraining S3D on Kinetics 400. Ignored if ``video_pretrained`` is ``False``.
        proj_out_dim (int): output dimension to project both encoders' outputs to.
            Defaults to ``256``, the value used by MUGEN.
        proj_dropout (float): dropout probability in the projection layers.
            Defaults to ``0.1``, the value used by MUGEN.

    Returns:
        CLIP: CLIP model using MUGEN's video encoder and text encoder.

    """
    text_model = TextEncoder(
        model_config=text_model_config,
        padding_value=text_padding_value,
    )
    if text_pretrained:
        print(f"Loading pretrained DistilBERT from {text_model_name}.")
        text_model.model = DistilBertModel.from_pretrained(text_model_name)
    if text_pretrained and not text_trainable:
        # check `text_pretrained` because if model isn't pretrained, then it should be trainable
        for p in text_model.model.parameters():
            p.requires_grad = False
    elif not text_trainable:
        warnings.warn("`text_trainable` acts as True when `text_pretrained` is False.")

    text_encoder = nn.Sequential(
        text_model,
        Projection(text_model.out_dim, out_dim=proj_out_dim, dropout_prob=proj_dropout),
    )

    video_model = VideoEncoder()
    if video_pretrained:
        print(f"Loading pretrained video encoder from {video_pretrain_path}.")
        load_module_from_url(video_model, video_pretrain_path)
    if video_pretrained and not video_trainable:
        # check `video_pretrained` because if model isn't pretrained, then it should be trainable
        for p in video_model.model.parameters():
            p.requires_grad = False
    elif not video_trainable:
        warnings.warn(
            "`video_trainable` acts as True when `video_pretrained` is False."
        )

    video_encoder = nn.Sequential(
        video_model,
        Projection(
            video_model.out_dim, out_dim=proj_out_dim, dropout_prob=proj_dropout
        ),
    )

    return CLIP(encoder_a=text_encoder, encoder_b=video_encoder)
