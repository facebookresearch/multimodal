# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import OrderedDict

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.modules.encoders.albef_vision_encoder import ALBEFVisionEncoder


class TestALBEFVisionEncoder:
    set_rng_seed(0)
    input = torch.randn(1, 3, 4, 4)
    proj_input = torch.randn(1, 1, 3)
    proj_state_dict = OrderedDict(
        [("weight", torch.randn(3, 3, 4, 4)), ("bias", torch.randn(3))]
    )
    attention_state_dict = OrderedDict(
        [
            ("in_proj_weight", torch.randn(9, 3)),
            ("in_proj_bias", torch.randn(9)),
            ("out_proj.weight", torch.randn(3, 3)),
            ("out_proj.bias", torch.randn(3)),
        ]
    )
    mlp_state_dict = OrderedDict(
        [
            ("0.weight", torch.randn(6, 3)),
            ("0.bias", torch.randn(6)),
            ("3.weight", torch.randn(3, 6)),
            ("3.bias", torch.randn(3)),
        ]
    )
    encoder_block_state_dict = OrderedDict(
        [
            ("ln_1.weight", torch.randn(3)),
            ("ln_1.bias", torch.randn(3)),
            ("ln_2.weight", torch.randn(3)),
            ("ln_2.bias", torch.randn(3)),
        ]
        + [("self_attention." + key, val) for key, val in attention_state_dict.items()]
        + [("mlp." + key, val) for key, val in mlp_state_dict.items()]
    )
    vit_state_dict = OrderedDict(
        [
            ("class_token", torch.randn(1, 1, 3)),
            ("encoder.pos_embedding", torch.randn(1, 2, 3)),
            ("encoder.ln.weight", torch.randn(3)),
            ("encoder.ln.bias", torch.randn(3)),
        ]
        + [("conv_proj." + key, val) for key, val in proj_state_dict.items()]
        + [
            ("encoder.layers.encoder_layer_0." + key, val)
            for key, val in encoder_block_state_dict.items()
        ]
        + [
            ("encoder.layers.encoder_layer_1." + key, val)
            for key, val in encoder_block_state_dict.items()
        ]
    )
    vision_encoder = ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=3,
    )

    def test_conv_proj(self):
        conv_proj = self.vision_encoder.conv_proj
        conv_proj.load_state_dict(self.proj_state_dict)
        output = conv_proj(self.input)
        expected = Tensor([-5.115712, 3.110137, -2.686451]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_attention(self):
        attention = self.vision_encoder.encoder.layers[0].self_attention
        attention.load_state_dict(self.attention_state_dict)
        output = attention(self.proj_input)
        expected = Tensor([0.799757, 0.632195, 2.890842]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_mlp_block(self):
        mlp = self.vision_encoder.encoder.layers[0].mlp
        mlp.load_state_dict(self.mlp_state_dict)
        output = mlp(self.proj_input)
        expected = Tensor([14.869835, 3.666760, 0.993663]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_encoder_block(self):
        encoder_block = self.vision_encoder.encoder.layers[0]
        encoder_block.load_state_dict(self.encoder_block_state_dict)
        output = encoder_block(self.proj_input)
        expected = Tensor([41.667698, 15.369812, 1.672322]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_vision_transformer(self):
        vit = self.vision_encoder
        vit.load_state_dict(self.vit_state_dict)
        output = vit(self.input)
        expected = Tensor(
            [
                [-2.992489e-02, -3.361803e-01, 2.282364e00],
                [-1.769625e-03, -3.862467e-01, 2.380805e00],
            ]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)
