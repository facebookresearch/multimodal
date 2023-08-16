# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerOutput,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(4)


class TestTransformerEncoderLayer:
    @pytest.fixture
    def get_encoder_layer(self):
        def create_layer(norm_first):
            model = TransformerEncoderLayer(
                d_model=2,
                n_head=1,
                dim_feedforward=2,
                norm_first=norm_first,
            )
            init_weights_with_constant(model)
            model.eval()
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return Tensor([[[1, 2], [4, 2]]])

    @pytest.mark.parametrize(
        "norm_first, expected_output",
        [
            (True, Tensor([[[15.0, 16.0], [18.0, 16.0]]])),
            (False, Tensor([[[0.0, 2.0], [2.0, 0.0]]])),
        ],
    )
    def test_forward(self, norm_first, expected_output, inputs, get_encoder_layer):
        model = get_encoder_layer(norm_first)
        actual = model(inputs)
        assert_expected(actual, expected_output, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "norm_first",
        [(True,), (False,)],
    )
    def test_scripting(self, norm_first, inputs, get_encoder_layer):
        model = get_encoder_layer(norm_first)
        scripted_model = torch.jit.script(model)
        assert_expected(scripted_model(inputs), model(inputs), rtol=0, atol=1e-4)


class TestTransformerEncoder:
    @pytest.fixture
    def get_encoder(self):
        def create_encoder(norm_first, final_layer_norm_eps=None):
            model = TransformerEncoder(
                n_layer=2,
                d_model=2,
                n_head=1,
                dim_feedforward=2,
                norm_first=norm_first,
                final_layer_norm_eps=final_layer_norm_eps,
            )
            init_weights_with_constant(model)
            model.eval()
            return model

        return create_encoder

    @pytest.fixture
    def inputs(self):
        return Tensor([[[2, 3], [1, 2]]])

    @pytest.mark.parametrize(
        "norm_first, return_hidden_states, expected_output",
        [
            (
                True,
                False,
                TransformerOutput(
                    last_hidden_state=Tensor([[[30.0, 31.0], [29.0, 30.0]]])
                ),
            ),
            (
                False,
                False,
                TransformerOutput(last_hidden_state=Tensor([[[0.0, 2.0], [0.0, 2.0]]])),
            ),
            (
                True,
                True,
                TransformerOutput(
                    last_hidden_state=Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    hidden_states=[
                        Tensor([[[16.0, 17.0], [15.0, 16.0]]]),
                        Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    ],
                ),
            ),
            (
                False,
                True,
                TransformerOutput(
                    last_hidden_state=Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    hidden_states=[
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    ],
                ),
            ),
        ],
    )
    def test_forward(
        self, norm_first, return_hidden_states, expected_output, inputs, get_encoder
    ):
        model = get_encoder(norm_first)
        actual = model(inputs, return_hidden_states=return_hidden_states)
        if expected_output.hidden_states is None:
            assert actual.hidden_states is None
        else:
            assert_expected(actual.hidden_states[0], inputs)
            for state_1, state_2 in zip(
                expected_output.hidden_states, actual.hidden_states[1:]
            ):
                assert_expected(state_1, state_2)

        assert actual.attentions == expected_output.attentions
        assert_expected(
            actual.last_hidden_state,
            expected_output.last_hidden_state,
            rtol=0,
            atol=1e-4,
        )

    @pytest.mark.parametrize(
        "norm_first, expected_output",
        [
            (
                True,
                TransformerOutput(
                    last_hidden_state=Tensor([[[1.9073e-05, 2.0], [2.2888e-05, 2.0]]]),
                    hidden_states=[
                        Tensor([[[16.0, 17.0], [15.0, 16.0]]]),
                        Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    ],
                ),
            ),
            (
                False,
                TransformerOutput(
                    last_hidden_state=Tensor([[[5.0068e-06, 2.0], [5.0068e-06, 2.0]]]),
                    hidden_states=[
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    ],
                ),
            ),
        ],
    )
    def test_forward_with_final_ln(
        self, norm_first, expected_output, inputs, get_encoder
    ):
        model = get_encoder(norm_first=norm_first, final_layer_norm_eps=1e-5)
        actual = model(inputs, return_hidden_states=True)
        assert_expected(
            expected_output.last_hidden_state,
            actual.last_hidden_state,
            rtol=0,
            atol=1e-4,
        )
        for state_1, state_2 in zip(
            expected_output.hidden_states, actual.hidden_states[1:]
        ):
            assert_expected(state_1, state_2)

    @pytest.mark.parametrize(
        "norm_first",
        [(True,), (False,)],
    )
    def test_scripting(self, norm_first, inputs, get_encoder):
        model = get_encoder(norm_first)
        scripted_model = torch.jit.script(model)
        assert_expected(scripted_model(inputs), model(inputs), rtol=0, atol=1e-4)
