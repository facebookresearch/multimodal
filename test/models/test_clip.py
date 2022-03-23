# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.architectures.clip import CLIPArchitecture
from torchmultimodal.modules.encoders.clip_resnet_encoder import ResNetForCLIP
from torchmultimodal.modules.encoders.clip_text_encoder import CLIPTextEncoder
from torchmultimodal.utils.common import get_current_device
from torchvision.models.vision_transformer import VisionTransformer


class TestCLIPModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.device = get_current_device()
        self.context_length = 77

    def test_clip_resnet_forward(self):
        resnet_encoder = ResNetForCLIP(
            layers=(3, 4, 6, 3),
            output_dim=12,
            heads=10,
            width=20,
        )
        text_encoder = CLIPTextEncoder(
            embedding_dim=12,
            context_length=self.context_length,
            vocab_size=100,
            width=512,
            heads=8,
            layers=12,
        )
        clip_resnet = CLIPArchitecture(
            vision_encoder=resnet_encoder,
            text_encoder=text_encoder,
        )
        clip_resnet = clip_resnet.to(self.device)
        self.assertTrue(isinstance(clip_resnet, torch.nn.Module))

        text = torch.randint(1, 79, (self.context_length,), dtype=torch.long).unsqueeze(
            0
        )
        image = torch.randn(3, 224, 224).unsqueeze(0)

        clip_resnet_scores = clip_resnet(image=image, text=text)
        self.assertEqual(clip_resnet_scores["image"].size(), torch.Size((1, 12)))
        self.assertEqual(clip_resnet_scores["text"].size(), torch.Size((1, 12)))

    def test_clip_vit_forward(self):
        vit_encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=12,
        )
        text_encoder = CLIPTextEncoder(
            embedding_dim=12,
            context_length=self.context_length,
            vocab_size=100,
            width=512,
            heads=8,
            layers=12,
        )

        text = torch.randint(1, 79, (self.context_length,), dtype=torch.long).unsqueeze(
            0
        )
        image = torch.randn(3, 224, 224).unsqueeze(0)
        clip_vit = CLIPArchitecture(
            vision_encoder=vit_encoder, text_encoder=text_encoder
        )
        clip_vit = clip_vit.to(self.device)
        self.assertTrue(isinstance(clip_vit, torch.nn.Module))

        clip_vit_scores = clip_vit(image=image, text=text)
        self.assertEqual(clip_vit_scores["image"].size(), torch.Size((1, 12)))
        self.assertEqual(clip_vit_scores["text"].size(), torch.Size((1, 12)))
