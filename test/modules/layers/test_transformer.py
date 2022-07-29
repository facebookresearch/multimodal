# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, init_params_ones, set_rng_seed
from torchmultimodal.modules.layers.transformer import (
    _apply_layernorm,
    FLAVATransformerEncoder,
    TransformerEncoderCrossAttentionLayer,
    TransformerEncoderLayer,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestFLAVATransformerEncoder:
    @pytest.fixture
    def encoder(self):
        return FLAVATransformerEncoder(
            hidden_size=2, num_attention_heads=2, num_hidden_layers=2
        )

    @pytest.fixture
    def inputs(self):
        return torch.rand((2, 3, 2))

    def test_flava_encoder_forward(self, encoder, inputs):
        output = encoder(inputs)

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


class TestTransformerEncoderLayer:
    @pytest.fixture
    def get_encoder_layer(self):
        def create_layer(norm_first):
            model = TransformerEncoderLayer(2, 1, 2, norm_first=norm_first)
            init_params_ones(model)
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    def test_attention_block(self, inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual, _ = model._attention_block(inputs)
        expected = model.attention(inputs)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_feedforward_block(self, inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model._feedforward_block(inputs)
        expected = model.output(
            model.intermediate_activation(model.intermediate(inputs))
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward_prenorm(self, inputs, get_encoder_layer):
        model = get_encoder_layer(True)
        actual = model(inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [[13.0586, 15.2632], [13.8162, 14.1505]],
                        [[14.1075, 13.7220], [11.3979, 14.6245]],
                    ],
                    [
                        [[13.1316, 13.7949], [14.3976, 14.6699]],
                        [[13.9463, 14.0467], [12.2329, 11.8795]],
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
                        [[-2.3842e-07, 2.0000e00], [-7.7486e-06, 2.0000e00]],
                        [[2.0000e00, 1.8477e-06], [-1.1921e-07, 2.0000e00]],
                    ],
                    [
                        [[9.5367e-07, 2.0000e00], [-1.9073e-06, 2.0000e00]],
                        [[1.8477e-05, 2.0000e00], [2.0000e00, 5.9605e-08]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestTransformerEncoderCrossAttentionLayer:
    @pytest.fixture
    def get_encoder_layer(self):
        def create_layer(norm_first):
            model = TransformerEncoderCrossAttentionLayer(
                2, 1, 2, norm_first=norm_first
            )
            init_params_ones(model)
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    @pytest.fixture
    def cross_inputs(self):
        return torch.randn(1, 2, 2, 2, 2)

    def test_self_attention_block(self, inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model._self_attention_block(inputs)
        expected = model.attention(inputs)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_cross_attention_block(self, inputs, cross_inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model._cross_attention_block(inputs, cross_inputs)
        expected = model.cross_attention(inputs, cross_inputs)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_feedforward_block(self, inputs, get_encoder_layer):
        model = get_encoder_layer(False)
        actual = model._feedforward_block(inputs)
        expected = model.output(
            model.intermediate_activation(model.intermediate(inputs))
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_forward_prenorm(self, inputs, cross_inputs, get_encoder_layer):
        model = get_encoder_layer(True)
        actual = model(inputs, cross_inputs)
        expected = torch.tensor(
            [
                [
                    [
                        [[7.0000, 9.0000], [7.0000, 9.0000]],
                        [[9.0000, 7.0000], [7.0000, 9.0000]],
                    ],
                    [
                        [[7.0000, 9.0000], [7.0000, 9.0000]],
                        [[7.0000, 9.0000], [9.0000, 7.0000]],
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
                    [
                        [[0.0000e00, 2.0000e00], [3.5763e-07, 2.0000e00]],
                        [[2.0000e00, -2.3842e-07], [0.0000e00, 2.0000e00]],
                    ],
                    [
                        [[0.0000e00, 2.0000e00], [0.0000e00, 2.0000e00]],
                        [[0.0000e00, 2.0000e00], [2.0000e00, 0.0000e00]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_apply_layernorm():
    x = torch.ones(1, 1, dtype=torch.float16)
    norm = torch.nn.LayerNorm(1)
    output = _apply_layernorm(x, norm, mixed_precision=True)
    assert output.dtype == torch.float16
    # layer norm should not work with fp16
    with pytest.raises(RuntimeError):
        output = _apply_layernorm(x, norm, mixed_precision=False)
