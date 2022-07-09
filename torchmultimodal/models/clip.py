# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from torchmultimodal.modules.encoders.clip_resnet_encoder import ResNetForCLIP
from torchmultimodal.modules.encoders.clip_text_encoder import CLIPTextEncoder
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.models.vision_transformer import VisionTransformer


class CLIPOutput(NamedTuple):
    embeddings_a: torch.Tensor
    embeddings_b: torch.Tensor


class CLIP(nn.Module):
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


def clip_vit_b16():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,  # based on https://git.io/JMpJK
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    return CLIP(vision_encoder, text_encoder)


def clip_vit_b32():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    return CLIP(vision_encoder, text_encoder)


def clip_vit_l14():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=14,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=768,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=768, width=768, heads=12)
    return CLIP(vision_encoder, text_encoder)


def clip_rn50():
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=1024,
        width=2048,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    return CLIP(vision_encoder, text_encoder)


def clip_rn101():
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 23, 3),
        output_dim=1024,
        heads=1024,
        width=2048,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    return CLIP(vision_encoder, text_encoder)


# Note: these models require larger image sizes
def clip_rn50x4():
    vision_encoder = ResNetForCLIP(
        layers=(4, 6, 10, 6),
        output_dim=640,
        heads=1280,
        input_resolution=288,
        width=2560,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024, width=640, heads=12)
    return CLIP(vision_encoder, text_encoder)


def clip_rn50x16():
    vision_encoder = ResNetForCLIP(
        layers=(6, 8, 18, 8),
        output_dim=768,
        heads=1536,
        input_resolution=384,
        width=3072,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=768, width=768, heads=12)
    return CLIP(vision_encoder, text_encoder)


def clip_rn50x64():
    vision_encoder = ResNetForCLIP(
        layers=(3, 15, 36, 10),
        output_dim=1024,
        heads=2048,
        input_resolution=448,
        width=4096,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024, width=1024, heads=16)
    return CLIP(vision_encoder, text_encoder)


# Note: these models use torchvision's ResNet
def clip_rn50_tv():
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        num_classes=1024,
    )
    text_encoder = CLIPTextEncoder()
    return CLIP(vision_encoder, text_encoder)


def clip_rn101_tv():
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder()
    return CLIP(vision_encoder, text_encoder)
