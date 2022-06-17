# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_text_encoder import ALBEFTextEncoder


class TestALBEFTextEncoder:
    set_rng_seed(0)
    text_encoder = ALBEFTextEncoder(hidden_size=3, num_attention_heads=1)

    def test_text_encoder(self):
        set_rng_seed(0)
        input_ids = torch.randint(10, (2, 2))
        text_atts = torch.randn(2, 2)
        output = self.text_encoder(input_ids, text_atts)
        expected = Tensor(
            [
                [[-0.668618, -0.744909, 1.413527], [-0.643172, 1.412341, -0.769169]],
                [[-1.052131, -0.292326, 1.344457], [0.986411, 0.384430, -1.370841]],
            ]
        )
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_text_encoder_without_attention_mask(self):
        set_rng_seed(0)
        input_ids = torch.randint(10, (2, 2))
        output = self.text_encoder(input_ids)
        expected = Tensor(
            [
                [[-0.814115, -0.594398, 1.408513], [-0.712880, 1.414198, -0.701317]],
                [[-0.888834, -0.508200, 1.397035], [1.195881, 0.055820, -1.251700]],
            ]
        )
        assert_expected(output, expected, rtol=0, atol=1e-4)

    def test_invalid_input_length(self):
        input_ids = torch.randint(10, (2, 2, 3))
        text_atts = torch.randn(2, 2, 3)
        with pytest.raises(RuntimeError):
            self.text_encoder(input_ids, text_atts)

    def test_not_matching_attention_mask_shape(self):
        input_ids = torch.randint(10, (2, 2))
        text_atts = torch.randn(2, 3)
        with pytest.raises(RuntimeError):
            self.text_encoder(input_ids, text_atts)