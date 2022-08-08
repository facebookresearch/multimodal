# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.transformer import (
    TransformerCrossAttentionLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestTransformerEncoderLayer:
    @pytest.fixture
    def get_encoder_layer(self):
        def create_layer(norm_first):
            model = TransformerEncoderLayer(2, 1, 2, norm_first=norm_first)
            model.eval()
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    def test_forward_prenorm(self, inputs, get_encoder_layer):
        model = get_encoder_layer(True)
        actual = model(inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [[-1.5605, 2.3367], [-0.8028, 1.2239]],
                        [[-0.3491, 0.7343], [-3.2212, 1.6979]],
                    ],
                    [
                        [[-1.4874, 0.8684], [-0.2215, 1.7433]],
                        [[-0.6728, 1.1201], [-2.2237, -1.1081]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward_postnorm(self, inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model(inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [[-1.0000, 1.0000], [-1.0000, 1.0000]],
                        [[-1.0000, 1.0000], [-1.0000, 1.0000]],
                    ],
                    [
                        [[-1.0000, 1.0000], [-1.0000, 1.0000]],
                        [[-1.0000, 1.0000], [-1.0000, 1.0000]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestTransformerCrossAttentionLayer:
    @pytest.fixture
    def get_encoder_layer(self):
        def create_layer(norm_first):
            model = TransformerCrossAttentionLayer(2, 1, 2, norm_first=norm_first)
            model.eval()
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    @pytest.fixture
    def cross_inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    def test_forward_prenorm(self, inputs, cross_inputs, get_encoder_layer):
        model = get_encoder_layer(True)
        actual = model(inputs, cross_inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.5925, 1.1257], [-0.5925, 1.1257]],
                        [[-0.5925, 1.1257], [-0.5925, 1.1257]],
                    ],
                    [
                        [[-0.5925, 1.1257], [-0.5925, 1.1257]],
                        [[-0.5925, 1.1257], [-0.5925, 1.1257]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward_postnorm(self, inputs, cross_inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model(inputs, cross_inputs)
        expected = torch.tensor(
            [
                [
                    [[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]],
                    [[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestTransformerEncoder:
    @pytest.fixture
    def encoder(self):
        return TransformerEncoder(
            n_layer=2,
            d_model=2,
            n_head=2,
            dim_feedforward=3072,
            activation=nn.GELU,
            norm_first=True,
        )

    @pytest.fixture
    def inputs(self):
        return torch.rand((2, 3, 2))

    def test_forward(self, encoder, inputs):
        output = encoder(inputs, return_hidden_states=True, return_attn_weights=True)

        actual_last_hidden_state = output.last_hidden_state
        actual_hidden_states = torch.sum(torch.stack(output.hidden_states), dim=0)
        actual_attentions = torch.sum(torch.stack(output.attentions), dim=0)

        expected_last_hidden_state = torch.Tensor(
            [
                [
                    [0.4387462735, 2.2609438896],
                    [0.4937185347, 2.1975874901],
                    [0.1847360730, 2.4323265553],
                ],
                [
                    [0.4651480913, 2.1417639256],
                    [0.0404203087, 2.1411576271],
                    [-0.1758930087, 1.9571313858],
                ],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [[1.7473, 4.3043], [2.0372, 3.5570], [1.1414, 4.2733]],
                [[2.0239, 3.2944], [0.7497, 3.2925], [0.1008, 2.7405]],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [
                    [
                        [0.6837, 0.6326, 0.6837],
                        [0.6836, 0.6327, 0.6836],
                        [0.6837, 0.6326, 0.6837],
                    ],
                    [
                        [0.6090, 0.7821, 0.6090],
                        [0.5669, 0.8662, 0.5669],
                        [0.6090, 0.7821, 0.6090],
                    ],
                ],
                [
                    [
                        [0.6667, 0.6667, 0.6667],
                        [0.6667, 0.6667, 0.6667],
                        [0.6667, 0.6667, 0.6667],
                    ],
                    [
                        [0.6667, 0.6667, 0.6667],
                        [0.6667, 0.6667, 0.6667],
                        [0.6667, 0.6667, 0.6667],
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
