# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import set_rng_seed, assert_expected
from torchmultimodal.modules.layers.transformer import TransformerEncoder


class TestTransformer(unittest.TestCase):
    """
    Test the Transformer classes
    """

    def setUp(self):
        set_rng_seed(4)
        self.encoder = TransformerEncoder()
        self.test_input = torch.rand((3, 4, 768))

    def test_encoder(self):
        output = self.encoder(self.test_input)

        actual_last_hidden_state = torch.mean(
            output.last_hidden_state, 2
        )  # mean across hidden dim
        actual_hidden_states = torch.mean(
            torch.stack(output.hidden_states), (0, 3)
        )  # mean across hidden & stack dim
        actual_attentions = torch.mean(
            torch.stack(output.attentions), (0, 1, 2)
        )  # mean across all dims except last two, which should be 4x4

        expected_last_hidden_state = torch.Tensor(
            [
                [0.4657, 0.4493, 0.4593, 0.4590],
                [0.4784, 0.4553, 0.4645, 0.4720],
                [0.5625, 0.5720, 0.5698, 0.5460],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [0.4799, 0.4721, 0.4774, 0.4611],
                [0.4884, 0.4756, 0.4815, 0.4799],
                [0.5430, 0.5478, 0.5513, 0.5308],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [0.2504, 0.2495, 0.2482, 0.2519],
                [0.2533, 0.2497, 0.2461, 0.2509],
                [0.2524, 0.2492, 0.2475, 0.2508],
                [0.2523, 0.2486, 0.2486, 0.2505],
            ]
        )

        assert_expected(
            actual_last_hidden_state, expected_last_hidden_state, rtol=0.0, atol=1e-4
        )
        assert_expected(
            actual_hidden_states, expected_hidden_states, rtol=0.0, atol=1e-4
        )
        assert_expected(actual_attentions, expected_attentions, rtol=0.0, atol=1e-4)
