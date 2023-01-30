# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
from torchmultimodal.utils.common import load_module_from_url
from torchvision.models.resnet import Bottleneck, ResNet


class CLIPOutput(NamedTuple):
    embeddings_a: torch.Tensor
    embeddings_b: torch.Tensor


CLIP_MODEL_MAPPING = {
    "vit_b16": "https://download.pytorch.org/models/multimodal/clip/clip_vit_b16.pt",
    "vit_b32": "https://download.pytorch.org/models/multimodal/clip/clip_vit_b32.pt",
    "vit_l14": "https://download.pytorch.org/models/multimodal/clip/clip_vit_l14.pt",
    "rn50": "https://download.pytorch.org/models/multimodal/clip/clip_rn50.pt",
    "rn101": "https://download.pytorch.org/models/multimodal/clip/clip_rn101.pt",
    "rn50x4": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x4.pt",
    "rn50x16": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x16.pt",
    "rn50x64": "https://download.pytorch.org/models/multimodal/clip/clip_rn50x64.pt",
}


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
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

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


def clip_vit_b16(pretrained: bool = False) -> CLIP:
    vision_encoder = CLIPViTEncoder(
        image_size=224, patch_size=16, layers=12, heads=12, width=768, embedding_dim=512
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["vit_b16"])
    return clip


def clip_vit_b32(pretrained: bool = False) -> CLIP:
    vision_encoder = CLIPViTEncoder(
        image_size=224, patch_size=32, layers=12, heads=12, width=768, embedding_dim=512
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["vit_b32"])
    return clip


def clip_vit_l14(pretrained: bool = False) -> CLIP:
    vision_encoder = CLIPViTEncoder(
        image_size=224,
        patch_size=14,
        layers=24,
        heads=16,
        width=1024,
        embedding_dim=768,
    )
    text_encoder = CLIPTextEncoder(
        embedding_dim=768, width=768, dim_feedforward=3072, heads=12
    )
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["vit_l14"])
    return clip


def clip_rn50(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=32,
        width=64,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn50"])
    return clip


def clip_rn101(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 23, 3),
        output_dim=512,
        heads=32,
        width=64,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn101"])
    return clip


# Note: these models require larger image sizes
def clip_rn50x4(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(4, 6, 10, 6),
        output_dim=640,
        heads=40,
        input_resolution=288,
        width=80,
    )
    text_encoder = CLIPTextEncoder(
        embedding_dim=640, width=640, dim_feedforward=2560, heads=10
    )
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn50x4"])
    return clip


def clip_rn50x16(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(6, 8, 18, 8),
        output_dim=768,
        heads=48,
        input_resolution=384,
        width=96,
    )
    text_encoder = CLIPTextEncoder(
        embedding_dim=768, width=768, dim_feedforward=3072, heads=12
    )
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn50x16"])
    return clip


def clip_rn50x64(pretrained: bool = False) -> CLIP:
    vision_encoder = ResNetForCLIP(
        layers=(3, 15, 36, 10),
        output_dim=1024,
        heads=64,
        input_resolution=448,
        width=128,
    )
    text_encoder = CLIPTextEncoder(
        embedding_dim=1024, width=1024, dim_feedforward=4096, heads=16
    )
    clip = CLIP(vision_encoder, text_encoder)
    if pretrained:
        load_module_from_url(clip, CLIP_MODEL_MAPPING["rn50x64"])
    return clip


# Note: these models use torchvision's ResNet
def clip_rn50_tv() -> CLIP:
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        num_classes=1024,
    )
    text_encoder = CLIPTextEncoder()
    return CLIP(vision_encoder, text_encoder)


def clip_rn101_tv() -> CLIP:
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder()
    return CLIP(vision_encoder, text_encoder)
