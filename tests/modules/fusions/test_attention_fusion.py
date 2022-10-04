# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from tests.test_utils import assert_expected
from torchmultimodal.modules.fusions.attention_fusion import AttentionFusionModule


class TestAttentionFusionModule(unittest.TestCase):
    def setUp(self):
        self.channel_to_encoder_dim = {
            "channel_1": 3,
            "channel_2": 3,
            "channel_3": 4,
        }
        self.batch_size = 2
        self.input = {}
        for channel, dim in self.channel_to_encoder_dim.items():
            self.input[channel] = torch.rand((self.batch_size, dim))

    def test_no_projection_dim(self):
        fusion = AttentionFusionModule(self.channel_to_encoder_dim)
        fused = fusion(self.input)
        assert_expected(fused.size(), (self.batch_size, 3))

    def test_input_projection_dim(self):
        fusion = AttentionFusionModule(
            self.channel_to_encoder_dim, encoding_projection_dim=2
        )
        fused = fusion(self.input)
        assert_expected(fused.size(), (self.batch_size, 2))

    def test_scripted_model(self):
        fusion = AttentionFusionModule(
            self.channel_to_encoder_dim, encoding_projection_dim=2
        )
        scripted_model = torch.jit.script(fusion)
        fused = scripted_model(self.input)
        assert_expected(fused.size(), (self.batch_size, 2))
