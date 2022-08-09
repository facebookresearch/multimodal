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
        model = TransformerEncoder(
            n_layer=2,
            d_model=2,
            n_head=2,
            dim_feedforward=3072,
            activation=nn.GELU,
            norm_first=True,
        )
        model.eval()
        return model

    @pytest.fixture
    def inputs(self):
        return torch.rand((2, 3, 2))

    def test_forward(self, inputs, encoder):
        output = encoder(inputs, return_hidden_states=True, return_attn_weights=True)

        actual_last_hidden_state = output.last_hidden_state
        actual_hidden_states = torch.sum(torch.stack(output.hidden_states), dim=0)
        actual_attentions = torch.sum(torch.stack(output.attentions), dim=0)

        expected_last_hidden_state = torch.Tensor(
            [
                [[1.6669, 0.3613], [1.0610, 0.0896], [0.9768, -0.0814]],
                [[2.3306, 0.6623], [1.8439, 0.7909], [1.6566, -0.0360]],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [[3.4371, 0.9657], [1.7571, 0.0734], [1.5043, -0.4397]],
                [[5.1976, 1.9218], [3.8499, 2.2402], [3.1757, -0.1730]],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [
                    [
                        [0.8520, 0.5740, 0.5740],
                        [0.6232, 0.6884, 0.6884],
                        [0.6232, 0.6884, 0.6884],
                    ],
                    [
                        [0.5859, 0.7071, 0.7071],
                        [0.6515, 0.6742, 0.6742],
                        [0.6515, 0.6742, 0.6742],
                    ],
                ],
                [
                    [
                        [0.7392, 0.5216, 0.7392],
                        [0.6434, 0.7132, 0.6434],
                        [0.7392, 0.5216, 0.7392],
                    ],
                    [
                        [0.6207, 0.7586, 0.6207],
                        [0.6589, 0.6822, 0.6589],
                        [0.6207, 0.7586, 0.6207],
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

        # set flags to false
        output = encoder(inputs)
        actual_last_hidden_state = output.last_hidden_state
        assert_expected(
            actual_last_hidden_state, expected_last_hidden_state, rtol=0.0, atol=1e-4
        )
