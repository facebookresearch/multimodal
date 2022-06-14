# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import OrderedDict

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_vision_encoder import PatchEmbed


class TestALBEFVisionEncoder:
    def test_conv_proj(self):
        set_rng_seed(0)
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
        input = torch.randn(1, 3, 4, 4)
        output = conv_proj(input)
        expected = Tensor([14.094812, 13.545728, 14.223189]).reshape(1, 1, 3)
        assert_expected(output, expected)
