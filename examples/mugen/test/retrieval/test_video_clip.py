# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mugen.retrieval.video_clip import TextEncoder

from test.test_utils import assert_expected, set_rng_seed


class TestTextEncoder:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)
        text_input = ["raw text sample for testing encoder"]
        encoder = TextEncoder(device="cpu")
        return encoder, text_input

    def test_forward(self, start):
        encoder, text_input = start
        out = encoder(text_input)
        expected_sum = -5.8399
        assert_expected(actual=out.shape, expected=torch.Size([1, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )
