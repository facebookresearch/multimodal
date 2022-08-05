# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected
from torchmultimodal.modules.encoders.clip_resnet_encoder import ResNetForCLIP
from torchmultimodal.utils.common import get_current_device


class TestResnetEncoder:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        torch.manual_seed(1234)

    @pytest.fixture
    def device(self):
        return get_current_device()

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
