# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.text_encoder import text_encoder


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestTextEncoder:
    @pytest.fixture
    def encoder(self):
        return text_encoder(hidden_size=3, num_attention_heads=1)

    def test_forward(self, encoder):
        input_ids = torch.randint(10, (2, 2))
        text_atts = Tensor([[1, 1], [1, 0]])
        output = encoder(input_ids, text_atts)
        expected = Tensor(
            [
                [[-0.7098, -0.7044, 1.4142], [-0.5453, -0.8574, 1.4027]],
                [[-0.2194, -1.1002, 1.3196], [0.3886, -1.3719, 0.9833]],
            ]
        )
        assert_expected(output.last_hidden_state, expected, rtol=0, atol=1e-4)
