# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import OrderedDict

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.modules.encoders.albef_vision_encoder import (
    Attention,
    Mlp,
    PatchEmbed,
)


class TestALBEFVisionEncoder:
    def test_conv_proj(self):
        set_rng_seed(0)
        input = torch.randn(1, 3, 4, 4)
        state_dict = OrderedDict(
            [("proj.weight", torch.randn(3, 3, 4, 4)), ("proj.bias", torch.randn(3))]
        )
        conv_proj = PatchEmbed(
            img_size=4,
            patch_size=4,
            in_chans=3,
            embed_dim=3,
        )
        conv_proj.load_state_dict(state_dict)
        output = conv_proj(input)
        expected = Tensor([-3.718719, 2.117420, 6.098923]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_attention(self):
        set_rng_seed(0)
        input = torch.randn(1, 1, 3)
        state_dict = OrderedDict(
            [
                ("qkv.weight", torch.randn(9, 3)),
                ("qkv.bias", torch.randn(9)),
                ("proj.weight", torch.randn(3, 3)),
                ("proj.bias", torch.randn(3)),
            ]
        )
        attention = Attention(
            3,
            num_heads=1,
            qkv_bias=True,
        )
        attention.load_state_dict(state_dict)
        output = attention(input)
        expected = Tensor([-12.703959, -9.637766, 7.394966]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_mlp_block(self):
        set_rng_seed(0)
        input = torch.randn(1, 1, 3)
        state_dict = OrderedDict(
            [
                ("fc1.weight", torch.randn(6, 3)),
                ("fc1.bias", torch.randn(6)),
                ("fc2.weight", torch.randn(3, 6)),
                ("fc2.bias", torch.randn(3)),
            ]
        )
        mlp = Mlp(3, hidden_features=6, act_layer=nn.GELU)
        mlp.load_state_dict(state_dict)
        output = mlp(input)
        expected = Tensor([4.896436, -0.737119, 1.037403]).reshape(1, 1, 3)
        assert_expected(output, expected, rtol=0, atol=1e-4)
