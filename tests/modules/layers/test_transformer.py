# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.transformer import (
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
    def encoder_ln(self):
        model = TransformerEncoder(
            n_layer=2,
            d_model=4,
            n_head=2,
            dim_feedforward=3072,
            activation=nn.GELU,
            norm_first=True,
            final_layer_norm_eps=1e-12,
        )
        model.eval()
        return model

    @pytest.fixture
    def inputs(self):
        return torch.rand((2, 3, 2))

    @pytest.fixture
    def inputs_ln(self):
        return torch.rand((2, 3, 4))

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

    def test_forward_ln(self, inputs_ln, encoder_ln):
        output = encoder_ln(
            inputs_ln, return_hidden_states=True, return_attn_weights=True
        )

        actual_last_hidden_state = output.last_hidden_state
        actual_hidden_states = torch.sum(torch.stack(output.hidden_states), dim=0)
        actual_attentions = torch.sum(torch.stack(output.attentions), dim=0)

        expected_last_hidden_state = torch.Tensor(
            [
                [
                    [0.0670, -1.5311, 1.2704, 0.1937],
                    [-0.7999, -0.9243, 1.5761, 0.1481],
                    [0.1925, -0.6254, 1.5352, -1.1022],
                ],
                [
                    [0.6513, -1.6111, -0.0297, 0.9895],
                    [0.1316, -1.1576, 1.5417, -0.5156],
                    [-0.3600, -1.4460, 0.6302, 1.1758],
                ],
            ]
        )
        expected_hidden_states = torch.Tensor(
            [
                [
                    [1.4894, 1.2685, 1.7669, 0.8038],
                    [0.1563, 0.3721, 4.3070, 2.4121],
                    [1.6380, 2.0771, 2.3102, 0.4584],
                ],
                [
                    [2.8866, 2.0093, 2.8522, 3.0838],
                    [1.8855, 1.0953, 2.5921, 0.6673],
                    [1.8191, 1.5908, 2.8085, 2.3234],
                ],
            ]
        )
        expected_attentions = torch.Tensor(
            [
                [
                    [
                        [0.6653, 0.6376, 0.6971],
                        [0.7078, 0.5621, 0.7302],
                        [0.6506, 0.6943, 0.6551],
                    ],
                    [
                        [0.6333, 0.7897, 0.5770],
                        [0.7207, 0.7019, 0.5774],
                        [0.7285, 0.7195, 0.5520],
                    ],
                ],
                [
                    [
                        [0.6919, 0.7021, 0.6060],
                        [0.6274, 0.7462, 0.6264],
                        [0.7025, 0.7090, 0.5885],
                    ],
                    [
                        [0.5826, 0.6227, 0.7947],
                        [0.6855, 0.6174, 0.6971],
                        [0.7317, 0.6057, 0.6625],
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
        output = encoder_ln(inputs_ln)
        actual_last_hidden_state = output.last_hidden_state
        assert_expected(
            actual_last_hidden_state, expected_last_hidden_state, rtol=0.0, atol=1e-4
        )
