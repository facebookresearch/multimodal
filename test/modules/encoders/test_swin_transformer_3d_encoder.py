# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.swin_transformer_3d_encoder import (
    PatchEmbed3d,
    SwinTransformer3dEncoder,
)
from torchmultimodal.utils.common import get_current_device


class TestSwinTransformer3dEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = get_current_device()

        # Setup Encoder to test
        self.encoder = SwinTransformer3dEncoder(
            patch_size=(2, 4, 4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            stochastic_depth_prob=0.2,
            norm_layer=torch.nn.LayerNorm,
            patch_embed=PatchEmbed3d,
        ).to(self.device)

    def test_swin_transformer_3d_encoder(self):
        self.assertTrue(isinstance(self.encoder, torch.nn.Module))
        image = torch.randn(1, 3, 1, 112, 112)  # B C D H W

        scores = self.encoder(image)
        self.assertEqual(scores.size(), torch.Size([1, 768]))
        self.assertAlmostEqual(scores.abs().sum().item(), 277.638336, 3)

    def test_swin_transformer_3d_scripting(self):
        torch.jit.script(self.encoder)
