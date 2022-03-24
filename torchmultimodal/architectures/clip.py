# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn


class CLIPArchitecture(nn.Module):
    """CLIP is a model for contrastive image and text pretraining.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains the
    encoders, while the loss is implemented in ContrastiveLossWithTemperature.


    Args:   vision_encoder (nn.Module): Instantiated vision encoder.
                See e.g. ResNetForCLIP class.
            text_encoder (nn.Module): Instantiated text encoder.
                See CLIPTextEncoder class.

    Inputs: image (Tensor): Tensor containing image features.
            text (Tensor): Tensor containing text features.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:

        img_embeddings = self.vision_encoder(image)
        text_embeddings = self.text_encoder(text)
        img_embeddings = F.normalize(img_embeddings)
        text_embeddings = F.normalize(text_embeddings)
        return {"image": img_embeddings, "text": text_embeddings}
