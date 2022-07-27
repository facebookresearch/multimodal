# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.transformer import FLAVATransformerEncoder


class TestTransformer(unittest.TestCase):
    """
    Test the Transformer classes
    """

    def setUp(self):
        set_rng_seed(4)
        self.encoder = FLAVATransformerEncoder(
            hidden_size=2, num_attention_heads=2, num_hidden_layers=2
        )
        self.test_input = torch.rand((2, 3, 2))

    def test_flava_encoder_forward(self):
        output = self.encoder(self.test_input)

        actual_last_hidden_state = output.last_hidden_state
        actual_hidden_states = torch.stack(output.hidden_states)
        actual_attentions = torch.stack(output.attentions)

        expected_last_hidden_state = torch.Tensor(
            [
                [[0.4387, 2.2609], [0.4937, 2.1976], [0.1847, 2.4323]],
                [[0.4651, 2.1418], [0.0404, 2.1412], [-0.1759, 1.9571]],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [
                    [[0.5924, 0.9998], [0.7723, 0.3792], [0.4945, 0.6260]],
                    [[0.8161, 0.2282], [0.3914, 0.2276], [0.1751, 0.0436]],
                ],
                [
                    [[0.7162, 1.0436], [0.7712, 0.9802], [0.4622, 1.2150]],
                    [[0.7426, 0.9244], [0.3179, 0.9238], [0.1016, 0.7398]],
                ],
                [
                    [[0.4387, 2.2609], [0.4937, 2.1976], [0.1847, 2.4323]],
                    [[0.4651, 2.1418], [0.0404, 2.1412], [-0.1759, 1.9571]],
                ],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [
                    [
                        [
                            [0.3503, 0.2993, 0.3503],
                            [0.3503, 0.2994, 0.3503],
                            [0.3503, 0.2993, 0.3503],
                        ],
                        [
                            [0.2756, 0.4488, 0.2756],
                            [0.2336, 0.5329, 0.2336],
                            [0.2756, 0.4488, 0.2756],
                        ],
                    ],
                    [
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                    ],
                    [
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                        [
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                            [0.3333, 0.3333, 0.3333],
                        ],
                    ],
                ],
            ]
        )

        assert_expected(
            actual_last_hidden_state, expected_last_hidden_state, rtol=0.0, atol=1e-4
        )
        assert_expected(
            actual_hidden_states, expected_hidden_states, rtol=0.0, atol=1e-4
        )
        assert_expected(actual_attentions, expected_attentions, rtol=0.0, atol=1e-4)
