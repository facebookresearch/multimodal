# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class CLIPOutput(NamedTuple):
    embeddings_a: torch.Tensor
    embeddings_b: torch.Tensor


class CLIPArchitecture(nn.Module):
    """CLIP is a model for contrastive pretraining between two modalities.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains the
    encoders, while the loss is implemented in ContrastiveLossWithTemperature.


    Args:   encoder_a (nn.Module): Instantiated encoder for modality A.
                See e.g. ResNetForCLIP class.
            encoder_b (nn.Module): Instantiated encoder for modality B.
                See e.g. CLIPTextEncoder class.

    Inputs: features_a (Tensor): Tensor containing features of modality A.
            features_b (Tensor): Tensor containing features of modality B.
    """

    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> CLIPOutput:

        embeddings_a = self.encoder_a(features_a)
        embeddings_b = self.encoder_b(features_b)
        embeddings_a = F.normalize(embeddings_a)
        embeddings_b = F.normalize(embeddings_b)
        return CLIPOutput(embeddings_a=embeddings_a, embeddings_b=embeddings_b)
