# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.models.clip.image_encoder import CLIPViTEncoder, ResNetForCLIP
from torchmultimodal.utils.common import get_current_device


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(1234)


@pytest.fixture
def device():
    return get_current_device()


class TestResnetEncoder:
    def test_resnet(self, device):
        resnet = ResNetForCLIP(
            layers=(3, 4, 6, 3),
            output_dim=512,
            heads=1024,
        )

        assert isinstance(resnet, torch.nn.Module)
        image = torch.randn(3, 224, 224).unsqueeze(0)
        resnet = resnet.to(device)

        scores = resnet(image)
        assert_expected(actual=scores.size(), expected=torch.Size((1, 512)))
        assert_expected(actual=scores.sum().item(), expected=2.1351, rtol=0, atol=1e-3)


class TestCLIPViTEncoder:
    @pytest.fixture(autouse=True)
    def clip_vit_encoder(self):
        set_rng_seed(0)

        encoder = CLIPViTEncoder(
            embedding_dim=4,
            heads=2,
            layers=1,
            patch_size=2,
            image_size=16,
            width=2,
        )
        encoder.eval()
        return encoder

    def test_forward(self, clip_vit_encoder):
        input = torch.ones(2, 3, 16, 16)
        out = clip_vit_encoder(input)
        expected = Tensor(
            [[1.1296, -0.6523, 0.3949, -0.7351], [1.1296, -0.6523, 0.3949, -0.7351]]
        )
        assert_expected(expected, out, atol=1e-4, rtol=0)

    def test_invalid_input(self, clip_vit_encoder):
        input = torch.ones(2, 3, 5, 5)
        with pytest.raises(ValueError):
            clip_vit_encoder(input)

        input = torch.ones(2, 2, 16, 16)
        with pytest.raises(ValueError):
            clip_vit_encoder(input)
