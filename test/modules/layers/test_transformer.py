# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import set_rng_seed
from torchmultimodal.modules.layers.transformer import TransformerEncoder


class TestTransformer(unittest.TestCase):
    """
    Test the Transformer classes
    """

    def setUp(self):
        set_rng_seed(4)
        self.encoder = TransformerEncoder()
        self.test_input = torch.rand((5, 5, 768))

    def test_encoder(self):
        output = self.encoder(self.test_input)

        actual_last_hidden_state_mean = torch.mean(output.last_hidden_state).item()
        actual_hidden_states_mean = torch.mean(torch.cat(output.hidden_states)).item()
        actual_attentions_mean = torch.mean(torch.cat(output.attentions)).item()

        expected_last_hidden_state_mean = 0.4884
        expected_hidden_states_mean = 0.4951
        expected_attentions_mean = 0.2000

        self.assertAlmostEqual(
            actual_last_hidden_state_mean, expected_last_hidden_state_mean, places=4
        )
        self.assertAlmostEqual(
            actual_hidden_states_mean, expected_hidden_states_mean, places=4
        )
        self.assertAlmostEqual(
            actual_attentions_mean, expected_attentions_mean, places=4
        )
