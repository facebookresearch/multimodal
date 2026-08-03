# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from tests.test_utils import set_rng_seed
from torchmultimodal.models.albef.image_encoder import ALBEFVisionEncoder


class TestALBEFVisionEncoder:
    set_rng_seed(0)
    torch.set_printoptions(precision=6)
    vision_encoder = ALBEFVisionEncoder(
        image_size=4,
        patch_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        hidden_size=3,
        mlp_dim=6,
    )

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
