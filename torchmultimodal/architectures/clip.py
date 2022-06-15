# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Dict, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class CLIPArchitecture(nn.Module):
    """CLIP is a model for contrastive image and text pretraining.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains the
    encoders, while the loss is implemented in ContrastiveLossWithTemperature.


    Args:   encoders (nn.ModuleDict): Dict of instantiated encoders, keyed by modality.
                E.g. {"vision": ResNetForCLIP(), "text": CLIPTextEncoder()}

    Inputs: modalities (Dict[str, Tensor]): Dict of Tensor features, keyed by modality.
                Must contain one entry for every modality in ``encoders``.

    Output: CLIPOutput namedtuple with a field for every modality in ``encoders``.
    """

    def __init__(
        self,
        encoders: nn.ModuleDict,
    ):
        super().__init__()
        self.encoders = encoders

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
    ) -> NamedTuple:
        embeddings = {}
        for key, encoder in self.encoders.items():
            if key not in modalities:
                raise ValueError(f"{key} missing in input")
            embeddings[key] = F.normalize(encoder(modalities[key]))
        for key in modalities.keys():
            if key not in self.encoders:
                warnings.warn(f"Missing encoder for extra input {key}")

        clip_output = NamedTuple("CLIPOutput", **{k: torch.Tensor for k in self.encoders.keys()})  # type: ignore
        return clip_output(**embeddings)
