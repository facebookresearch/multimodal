# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.clip.image_encoder import ResNetForCLIP
from torchmultimodal.models.clip.model import CLIP
from torchmultimodal.models.clip.text_encoder import CLIPTextEncoder
from torchvision.models.vision_transformer import VisionTransformer


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(1234)


class TestCLIP:
    @pytest.fixture(scope="class")
    def context_length(self):
        return 77

    def test_clip_forward(self):
        encoder_a = torch.nn.Linear(5, 3)
        encoder_b = torch.nn.Linear(4, 3)
        clip = CLIP(encoder_a, encoder_b)

        input_a = torch.randint(1, 8, (2, 5), dtype=torch.float)
        input_b = torch.randint(1, 8, (2, 4), dtype=torch.float)

        assert isinstance(clip, torch.nn.Module)

        out = clip(input_a, input_b)
        assert (
            hasattr(out, "embeddings_a")
            and hasattr(out, "embeddings_b")
            and len(out) == 2
        )

        actual_a_embedding = out.embeddings_a
        actual_b_embedding = out.embeddings_b
        expected_a_embedding = torch.Tensor(
            [[-0.8066, -0.1749, 0.5647], [-0.7709, -0.1118, 0.6271]]
        )
        expected_b_embedding = torch.Tensor(
            [[-0.1719, 0.7932, 0.5842], [-0.2805, 0.8761, -0.3921]]
        )
        assert_expected(
            actual=actual_a_embedding, expected=expected_a_embedding, rtol=0, atol=1e-4
        )
        assert_expected(
            actual=actual_b_embedding, expected=expected_b_embedding, rtol=0, atol=1e-4
        )

    def test_clip_resnet_forward(self, context_length):
        resnet_encoder = ResNetForCLIP(
            layers=(3, 4, 6, 3),
            output_dim=12,
            heads=10,
            width=20,
        )
        text_encoder = CLIPTextEncoder(
            embedding_dim=12,
            context_length=context_length,
            vocab_size=100,
            width=512,
            heads=8,
            layers=12,
        )
        clip_resnet = CLIP(
            encoder_a=resnet_encoder,
            encoder_b=text_encoder,
        )

        assert isinstance(clip_resnet, torch.nn.Module)

        text = torch.randint(1, 79, (context_length,), dtype=torch.long).unsqueeze(0)
        image = torch.randn(3, 224, 224).unsqueeze(0)

        clip_resnet_scores = clip_resnet(features_a=image, features_b=text)
        assert_expected(
            torch.tensor(clip_resnet_scores.embeddings_a.size()), torch.tensor((1, 12))
        )
        assert_expected(
            torch.tensor(clip_resnet_scores.embeddings_b.size()), torch.tensor((1, 12))
        )

    def test_clip_vit_forward(self, context_length):
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
            context_length=context_length,
            vocab_size=100,
            width=512,
            heads=8,
            layers=12,
        )

        text = torch.randint(1, 79, (context_length,), dtype=torch.long).unsqueeze(0)
        image = torch.randn(3, 224, 224).unsqueeze(0)
        clip_vit = CLIP(encoder_a=vit_encoder, encoder_b=text_encoder)

        assert isinstance(clip_vit, torch.nn.Module)

        clip_vit_scores = clip_vit(features_a=image, features_b=text)
        assert_expected(
            torch.tensor(clip_vit_scores.embeddings_a.size()), torch.tensor((1, 12))
        )
        assert_expected(
            torch.tensor(clip_vit_scores.embeddings_b.size()), torch.tensor((1, 12))
        )
