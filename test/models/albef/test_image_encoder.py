# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.albef.image_encoder import ALBEFVisionEncoder


class TestALBEFVisionEncoder:
    set_rng_seed(0)
    torch.set_printoptions(precision=6)
    vision_encoder = ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_layers=2,
        num_heads=1,
        hidden_dim=3,
        mlp_dim=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    def test_vision_transformer(self):
        set_rng_seed(0)
        vit = self.vision_encoder
        input = torch.randn(1, 3, 4, 4)
        output = vit(input)
        expected = Tensor(
            [
                [1.399478, -0.875986, -0.523492],
                [-0.869867, 1.400589, -0.530722],
            ]
        ).unsqueeze(0)
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_invalid_input_length(self):
        input = torch.randn(3, 4, 4)
        with pytest.raises(IndexError, match="index out of range"):
            self.vision_encoder(input)

    def test_invalid_image_channel_dim(self):
        input = torch.rand(1, 1, 4, 4)
        with pytest.raises(RuntimeError, match="channels"):
            self.vision_encoder(input)

    def test_invalid_image_height(self):
        input = torch.rand(1, 3, 5, 4)
        with pytest.raises(AssertionError, match="Wrong image height!"):
            self.vision_encoder(input)

    def test_invalid_image_width(self):
        input = torch.rand(1, 3, 4, 3)
        with pytest.raises(AssertionError, match="Wrong image width!"):
            self.vision_encoder(input)
