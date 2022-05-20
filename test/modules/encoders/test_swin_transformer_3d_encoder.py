# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.swin_transformer_3d_encoder import (
    PatchEmbed3d,
    PatchMerging3d,
    ShiftedWindowAttention3d,
    SwinTransformer3dEncoder,
)
from torchmultimodal.utils.common import get_current_device
from ...test_utils import set_rng_seed


class TestSwinTransformer3dEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(42)
        self.device = get_current_device()

        # Setup Encoder to test
        self.encoder = (
            SwinTransformer3dEncoder(
                patch_size=(2, 4, 4),
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=(8, 7, 7),
                stochastic_depth_prob=0.2,
                norm_layer=torch.nn.LayerNorm,
                patch_embed=PatchEmbed3d,
            )
            .to(self.device)
        )

    def test_swin_transformer_3d_encoder(self):
        image = torch.randn(1, 3, 1, 112, 112)  # B C D H W

        scores = self.encoder(image)
        self.assertEqual(scores.size(), torch.Size([1, 768]))
        self.assertAlmostEqual(scores.abs().sum().item(), 257.66665, 3)

    def test_swin_transformer_3d_scripting(self):
        torch.jit.script(self.encoder)


class TestSwinTransformer3dComponents(unittest.TestCase):
    def setUp(self):
        set_rng_seed(42)
        self.device = get_current_device()

    def test_patch_merging_3d(self):
        module = (
            PatchMerging3d(dim=12, norm_layer=torch.nn.LayerNorm).to(self.device)
        )
        x_in = torch.randn(1, 1, 56, 56, 12)
        x_out = module(x_in)

        self.assertEqual(x_out.size(), torch.Size([1, 1, 28, 28, 24]))
        self.assertAlmostEqual(x_out.abs().sum().item(), 8705.25390, 3)

    def test_shifted_window_attention_3d(self):
        module = (
            ShiftedWindowAttention3d(
                dim=12, window_size=(8, 7, 7), shift_size=(4, 3, 3), num_heads=3
            )
            .to(self.device)
        )
        x_in = torch.randn(1, 1, 56, 56, 12)
        x_out = module(x_in)

        self.assertEqual(x_out.size(), torch.Size([1, 1, 56, 56, 12]))
        self.assertAlmostEqual(x_out.abs().sum().item(), 6189.71777, 3)

    def test_shifted_window_attention_3d_zero_shift(self):
        module = (
            ShiftedWindowAttention3d(
                dim=12, window_size=(8, 7, 7), shift_size=(0, 0, 0), num_heads=3
            )
            .to(self.device)
        )
        x_in = torch.randn(1, 1, 56, 56, 12)
        x_out = module(x_in)

        self.assertEqual(x_out.size(), torch.Size([1, 1, 56, 56, 12]))
        self.assertAlmostEqual(x_out.abs().sum().item(), 6131.83691, 3)
