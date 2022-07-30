# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.image_encoder import mdetr_resnet101_backbone


class TestMDETRImageEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.test_tensor = torch.rand(4, 3, 64, 64)
        self.mask = torch.zeros(4, 64, 64)
        self.resnet101_encoder = mdetr_resnet101_backbone()
        self.resnet101_encoder.eval()

    def test_resnet_101_forward(self):
        # Taken from [:, 2, :, :] of forward outputs from
        # MDETR backbone with pretrained ImageNetV1 weights
        expected = torch.Tensor(
            [
                [[0.4230, 0.9407], [0.8498, 0.5046]],
                [[1.1702, 1.6584], [1.4689, 1.7062]],
                [[1.3003, 1.7222], [2.2372, 1.8877]],
                [[1.5309, 2.1169], [1.6040, 1.6911]],
            ]
        )
        # Get corresponding slice from last layer of outputs
        out, _ = self.resnet101_encoder(self.test_tensor, self.mask)
        actual = out[:, 2, :, :]
        self.assertEqual(out.size(), (4, 2048, 2, 2))
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)
