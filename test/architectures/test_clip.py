# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.architectures.clip import CLIPArchitecture


class TestCLIPArchitecture:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)

        encoder_a = torch.nn.Linear(5, 3)
        encoder_b = torch.nn.Linear(4, 3)
        clip = CLIPArchitecture(encoder_a, encoder_b)

        input_a = torch.randint(1, 8, (2, 5), dtype=torch.float)
        input_b = torch.randint(1, 8, (2, 4), dtype=torch.float)

        return clip, input_a, input_b

    def test_forward(self, start):
        clip, input_a, input_b = start
        assert isinstance(clip, torch.nn.Module)

        out = clip(input_a, input_b)
        assert (
            hasattr(out, "embeddings_a")
            and hasattr(out, "embeddings_b")
            and len(out) == 2
        )

        actual_a_embedding = out.embeddings_a
        actual_b_embedding = out.embeddings_b
        expected_a_embedding = torch.Tensor(
            [[-0.8066, -0.1749, 0.5647], [-0.7709, -0.1118, 0.6271]]
        )
        expected_b_embedding = torch.Tensor(
            [[-0.1719, 0.7932, 0.5842], [-0.2805, 0.8761, -0.3921]]
        )
        assert_expected(
            actual=actual_a_embedding, expected=expected_a_embedding, rtol=0, atol=1e-4
        )
        assert_expected(
            actual=actual_b_embedding, expected=expected_b_embedding, rtol=0, atol=1e-4
        )
