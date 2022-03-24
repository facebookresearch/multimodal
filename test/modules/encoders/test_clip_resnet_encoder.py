# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.clip_resnet_encoder import ResNetForCLIP
from torchmultimodal.utils.common import get_current_device


class TestCLIPModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.device = get_current_device()

    def test_resnet(self):
        resnet = ResNetForCLIP(
            layers=(3, 4, 6, 3),
            output_dim=512,
            heads=1024,
        )

        self.assertTrue(isinstance(resnet, torch.nn.Module))
        image = torch.randn(3, 224, 224).unsqueeze(0)
        resnet = resnet.to(self.device)

        scores = resnet(image)
        self.assertEqual(scores.size(), torch.Size((1, 512)))
        self.assertAlmostEqual(scores.sum().item(), 2.1351, 3)
