# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchmultimodal.modules.fusions.deepset_fusion import (
    deepset_transformer,
    DeepsetFusionModule,
    DeepsetFusionWithTransformer,
)
from torchmultimodal.modules.layers.mlp import MLP


class TestDeepSetFusionModule(unittest.TestCase):
    def setUp(self):
        self.channel_to_encoder_dim = {
            "channel_1": 3,
            "channel_2": 3,
            "channel_3": 3,
        }
        self.batch_size = 2
        self.input = {}
        self.input_bsz_1 = {}
        for channel, dim in self.channel_to_encoder_dim.items():
            self.input[channel] = torch.rand((self.batch_size, dim))
            self.input_bsz_1[channel] = torch.rand((1, dim))

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(3, 3, batch_first=True),
            num_layers=1,
            norm=nn.LayerNorm(3),
        )
        self.mlp = MLP(in_dim=3, out_dim=4)

    def _do_assertions(self, fusion):
        fused = fusion(self.input)
        self.assertEqual(fused.size(), (self.batch_size, 4))

        fused_bsz_1 = fusion(self.input_bsz_1)
        self.assertEqual(fused_bsz_1.size(), (1, 4))

    def test_deepset_sum(self):
        fusion = DeepsetFusionModule(self.channel_to_encoder_dim, self.mlp, torch.sum)
        self._do_assertions(fusion)

    def test_deepset_mean(self):
        fusion = DeepsetFusionModule(self.channel_to_encoder_dim, self.mlp, torch.mean)
        self._do_assertions(fusion)

    def test_deepset_median(self):
        fusion = DeepsetFusionModule(
            self.channel_to_encoder_dim, self.mlp, torch.median
        )
        self._do_assertions(fusion)

    def test_deepset_min(self):
        fusion = DeepsetFusionModule(self.channel_to_encoder_dim, self.mlp, torch.min)
        self._do_assertions(fusion)

    def test_deepset_max(self):
        fusion = DeepsetFusionModule(self.channel_to_encoder_dim, self.mlp, torch.max)
        self._do_assertions(fusion)

    def test_deepset_invalid_pooling(self):
        def random(x, dim):
            return "random"

        fusion = DeepsetFusionModule(self.channel_to_encoder_dim, self.mlp, random)
        with self.assertRaises(ValueError):
            fusion(self.input)

    def test_deepset_auto_mapping(self):
        fusion = DeepsetFusionModule(
            self.channel_to_encoder_dim,
            self.mlp,
            torch.sum,
            modality_normalize=True,
            use_auto_mapping=True,
        )
        self._do_assertions(fusion)

    def test_deepset_modality_normalize(self):
        fusion = DeepsetFusionModule(
            self.channel_to_encoder_dim,
            self.mlp,
            torch.sum,
            modality_normalize=True,
        )
        self._do_assertions(fusion)

    def test_deepset_apply_attention(self):
        fusion = DeepsetFusionModule(
            self.channel_to_encoder_dim,
            self.mlp,
            torch.sum,
            modality_normalize=True,
            apply_attention=True,
        )
        self._do_assertions(fusion)

    def test_deepset_transformer(self):
        fusion = DeepsetFusionWithTransformer(
            self.channel_to_encoder_dim,
            self.mlp,
            self.transformer,
        )
        self._do_assertions(fusion)

    def test_torchscript(self):
        fusion = DeepsetFusionWithTransformer(
            self.channel_to_encoder_dim,
            self.mlp,
            self.transformer,
        )
        torch.jit.script(fusion)

        fusion = DeepsetFusionModule(
            self.channel_to_encoder_dim,
            self.mlp,
            torch.sum,
        )
        torch.jit.script(fusion)

    def test_get_deepset_transformer(self):
        fusion = deepset_transformer(
            self.channel_to_encoder_dim,
            self.mlp,
            num_transformer_att_heads=3,
        )
        self.assertTrue(isinstance(fusion, DeepsetFusionModule))
        self.assertTrue(isinstance(fusion.pooling_function, nn.TransformerEncoder))
