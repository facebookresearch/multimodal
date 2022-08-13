# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.clip_vit_encoder import CLIPViTEncoder


class TestCLIPViTEncoder:
    @pytest.fixture(autouse=True)
    def clip_vit_encoder(self):
        set_rng_seed(0)

        encoder = CLIPViTEncoder(
            embedding_dim=4,
            heads=2,
            layers=1,
            patch_size=2,
            image_size=16,
            width=2,
        )
        encoder.eval()
        return encoder

    def test_forward(self, clip_vit_encoder):
        input = torch.ones(2, 3, 16, 16)
        out = clip_vit_encoder(input)
        expected = Tensor(
            [[1.1296, -0.6523, 0.3949, -0.7351], [1.1296, -0.6523, 0.3949, -0.7351]]
        )
        assert_expected(expected, out, atol=1e-4, rtol=0)

    def test_invalid_input(self, clip_vit_encoder):
        input = torch.ones(2, 3, 5, 5)
        with pytest.raises(ValueError):
            clip_vit_encoder(input)

        input = torch.ones(2, 2, 16, 16)
        with pytest.raises(ValueError):
            clip_vit_encoder(input)
