# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.modules.encoders.albef_vision_encoder import ALBEFVisionEncoder


class TestALBEFVisionEncoder:
    set_rng_seed(0)
    vision_encoder = ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    def test_conv_proj(self):
        set_rng_seed(0)
        conv_proj = self.vision_encoder.conv_proj
        input = torch.randn(1, 3, 4, 4)
        output = conv_proj(input)
        expected = Tensor([-0.792905, -0.971702, -0.393140]).reshape(1, 3, 1, 1)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_attention(self):
        set_rng_seed(0)
        attention = self.vision_encoder.encoder.layers[0].self_attention
        input = torch.randn(1, 1, 3)
        output, _ = attention(query=input, key=input, value=input)
        expected = Tensor([0.226826, 0.262802, -0.304190]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_mlp_block(self):
        set_rng_seed(0)
        mlp = self.vision_encoder.encoder.layers[0].mlp
        input = torch.randn(1, 1, 3)
        output = mlp(input)
        expected = Tensor([1.498866, -1.010315, 0.139188]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_encoder_block(self):
        set_rng_seed(0)
        encoder_block = self.vision_encoder.encoder.layers[0]
        input = torch.randn(1, 1, 3)
        output = encoder_block(input)
        expected = Tensor([2.685928, -0.826006, -2.173242]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_vision_transformer(self):
        set_rng_seed(0)
        vit = self.vision_encoder
        input = torch.randn(1, 3, 4, 4)
        output = vit(input)
        expected = Tensor(
            [
                [1.268159, -0.092086, -1.176073],
                [-0.988629, 1.370074, -0.381445],
            ]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)
