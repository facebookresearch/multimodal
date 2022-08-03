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
    transformer_encoder,
    TransformerCrossAttentionLayer,
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
        return transformer_encoder(
            n_layers=2,
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
                [[0.7518, 1.0348], [0.8214, 1.4777], [0.6917, 1.2366]],
                [[0.6661, 1.4557], [0.5292, 0.6997], [0.1917, 0.8434]],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [[2.1316, 2.5687], [2.3714, 3.9094], [1.9824, 3.1862]],
                [[2.1966, 2.6867], [1.4734, 1.5089], [0.3254, 2.4932]],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [
                    [
                        [0.6327, 0.6836, 0.6836],
                        [0.6326, 0.6837, 0.6837],
                        [0.6326, 0.6837, 0.6837],
                    ],
                    [
                        [0.8662, 0.5669, 0.5669],
                        [0.7821, 0.6090, 0.6090],
                        [0.7821, 0.6090, 0.6090],
                    ],
                ],
                [
                    [
                        [0.6488, 0.6488, 0.7024],
                        [0.6488, 0.6488, 0.7024],
                        [0.6487, 0.6487, 0.7025],
                    ],
                    [
                        [0.7435, 0.7435, 0.5131],
                        [0.7435, 0.7435, 0.5131],
                        [0.7159, 0.7159, 0.5683],
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
