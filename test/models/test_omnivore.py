# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torchmultimodal.models.omnivore as omnivore
from torchmultimodal.utils.common import get_current_device

from ..test_utils import set_rng_seed


class TestOmnivoreModel(unittest.TestCase):
    def setUp(self):
        set_rng_seed(42)
        self.device = get_current_device()

    def test_omnivore_swin_t_forward(self):
        model = omnivore.omnivore_swin_t().to(self.device)
        self.assertTrue(isinstance(model, torch.nn.Module))

        image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
        image_score = model(image, input_type="image")
        self.assertEqual(image_score.size(), torch.Size((1, 1000)))
        self.assertAlmostEqual(image_score.abs().sum().item(), 178.92318, 3)

        rgbd = torch.randn(1, 4, 1, 112, 112)
        rgbd_score = model(rgbd, input_type="rgbd")
        self.assertEqual(rgbd_score.size(), torch.Size((1, 19)))
        self.assertAlmostEqual(rgbd_score.abs().sum().item(), 3.39016, 3)

        video = torch.randn(1, 3, 4, 112, 112)
        video_score = model(video, input_type="video")
        self.assertEqual(video_score.size(), torch.Size((1, 400)))
        self.assertAlmostEqual(video_score.abs().sum().item(), 102.76638, 3)

    def test_omnivore_forward_wrong_input_type(self):
        model = omnivore.omnivore_swin_t().to(self.device)

        image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
        with self.assertRaises(AssertionError) as cm:
            _ = model(image, input_type="_WRONG_TYPE_")
            self.assertEqual(
                "Unsupported input_type: _WRONG_TYPE_, please use one of {'video', 'rgbd', 'image'}",
                str(cm.exception),
            )
