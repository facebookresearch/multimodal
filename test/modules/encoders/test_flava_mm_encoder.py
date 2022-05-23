# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.encoders.flava_mm_encoder import flava_multimodal_encoder


class TestFlavaMMEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.mm_encoder = flava_multimodal_encoder(
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=1,
            intermediate_size=2,
        )

    def test_mm_encoder(self):
        input = torch.ones(2, 3, 2)
        out = self.mm_encoder(input)
        torch.testing.assert_close(
            out.last_hidden_state,
            torch.Tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ]
            ),
        )
        assert_expected(
            out.attentions,
            (
                torch.Tensor(
                    (
                        [
                            [
                                [
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                ]
                            ],
                            [
                                [
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                    [0.2500, 0.2500, 0.2500, 0.2500],
                                ]
                            ],
                        ]
                    )
                ),
            ),
        )

        assert_expected(out.pooler_output, torch.Tensor([[0.0, 0.0], [0.0, 0.0]]))

        assert_expected(
            out.hidden_states,
            (
                torch.Tensor(
                    [
                        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    ]
                ),
                torch.Tensor(
                    [
                        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    ]
                ),
            ),
        )
