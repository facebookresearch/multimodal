# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.swin_transformer_3d_encoder import (
    SwinTransformer3dEncoder,
    PatchEmbedOmnivore,
)
from torchmultimodal.utils.common import get_current_device


class TestSwinTransformer3dEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = get_current_device()

        # Setup Encoder to test
        embed_dim = 96
        norm_layer = torch.nn.LayerNorm
        patch_embed = PatchEmbedOmnivore(
            patch_size=(2, 4, 4),
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.encoder = SwinTransformer3dEncoder(
            embed_dim=embed_dim,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8, 7, 7),
            stochastic_depth_prob=0.2,
            norm_layer=norm_layer,
            patch_embed=patch_embed,
        ).to(self.device)

    def test_swin_transformer_3d_encoder(self):
        self.assertTrue(isinstance(self.encoder, torch.nn.Module))
        image = torch.randn(1, 3, 1, 112, 112)  # B C D H W

        scores = self.encoder(image)
        self.assertEqual(scores.size(), torch.Size([1, 768]))
        self.assertAlmostEqual(scores.abs().sum().item(), 268.00668, 3)

    def test_swin_transformer_3d_scripting(self):
        torch.jit.script(self.encoder)
