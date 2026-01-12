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
    TransformerDecoder,
    TransformerDecoderLayer,
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


class TestTransformerDecoderLayer:
    @pytest.fixture
    def get_decoder_layer(self):
        def create_layer(d_model, norm_first, use_cross_attention, custom_init=False):
            model = TransformerDecoderLayer(
                d_model=d_model,
                n_head=1,
                dim_feedforward=2,
                norm_first=norm_first,
                use_cross_attention=use_cross_attention,
            )
            init_weights_with_constant(model)
            if custom_init:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "norm" in name:
                            param.copy_(
                                torch.arange(param.shape[0]).to(dtype=torch.float)
                            )

            model.eval()
            return model

        return create_layer

    @pytest.fixture
    def inputs(self):
        return Tensor([[[1, 2], [4, 2], [1, 1]]])

    @pytest.fixture
    def hidden_states(self):
        return Tensor([[[1, 2, 3, 4], [4, 2, 0, 2]]])

    @pytest.fixture
    def encoder_hidden_states(self):
        return Tensor([[[5, 6, 7, 8], [8, 9, 11, 12], [2, 1, 0, 2], [0, 0, 4, 4]]])

    @pytest.fixture
    def past_key(self):
        return Tensor([[[[0, 1, 2, 3], [4, 5, 6, 7]]]])

    @pytest.fixture
    def past_value(self):
        return Tensor([[[[7, 6, 5, 4], [3, 2, 1, 0]]]])

    @pytest.fixture
    @pytest.mark.parametrize(
        "norm_first, expected_output",
        [
            (
                True,
                (Tensor([[[15.0, 16.0], [18.0, 16.0], [15.0, 15.0]]]), (None, None)),
            ),
            (False, (Tensor([[[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]]]), (None, None))),
        ],
    )
    # Without cross-attention, results should match encoder layer
    def test_forward_without_cross_attn(
        self, norm_first, expected_output, inputs, get_decoder_layer
    ):
        model = get_decoder_layer(
            d_model=2, norm_first=norm_first, use_cross_attention=False
        )
        actual = model(inputs)
        assert_expected(actual, expected_output, rtol=0, atol=1e-3)

    @pytest.mark.parametrize(
        "norm_first, mask, expected_output",
        [
            (
                True,
                torch.BoolTensor([[False, True], [True, False]]),
                (
                    Tensor(
                        [
                            [
                                [207.6306, 208.6306, 209.6306, 210.6306],
                                [225.2317, 223.2317, 221.2317, 223.2317],
                            ]
                        ]
                    ),
                    None,
                ),
            ),
            (
                True,
                None,
                (
                    Tensor(
                        [
                            [
                                [236.8329, 237.8329, 238.8329, 239.8329],
                                [225.2317, 223.2317, 221.2317, 223.2317],
                            ]
                        ]
                    ),
                    None,
                ),
            ),
            (
                False,
                torch.BoolTensor([[False, True], [True, False]]),
                (
                    Tensor(
                        [
                            [
                                [0.0000, 0.2642, 1.7713, 8.0006],
                                [0.0000, 0.6952, 0.5130, 8.1252],
                            ]
                        ]
                    ),
                    None,
                ),
            ),
            (
                False,
                None,
                (
                    Tensor(
                        [
                            [
                                [0.0000, 0.2642, 1.7713, 8.0006],
                                [0.0000, 0.6952, 0.5130, 8.1252],
                            ]
                        ]
                    ),
                    None,
                ),
            ),
        ],
    )
    def test_forward_with_cross_attn(
        self,
        norm_first,
        mask,
        expected_output,
        hidden_states,
        encoder_hidden_states,
        get_decoder_layer,
    ):
        model = get_decoder_layer(
            d_model=4, norm_first=norm_first, use_cross_attention=True, custom_init=True
        )
        actual = model(hidden_states, encoder_hidden_states, attention_mask=mask)
        assert_expected(actual, expected_output, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "norm_first, expected_current_key_value",
        [
            (
                True,
                # pre-norm: 1s matrix squared + 1s bias
                Tensor([[[[5, 5, 5, 5], [5, 5, 5, 5]]]]),
            ),
            (
                False,
                # Should equal hidden_states * ones matrix weight + ones bias
                # (due to init_weights_with_constant)
                Tensor([[[[11, 11, 11, 11], [9, 9, 9, 9]]]]),
            ),
        ],
    )
    def test_kv_caching(
        self,
        norm_first,
        expected_current_key_value,
        hidden_states,
        encoder_hidden_states,
        past_key,
        past_value,
        get_decoder_layer,
    ):
        model = get_decoder_layer(
            d_model=4, norm_first=norm_first, use_cross_attention=True
        )
        actual = model(
            hidden_states,
            encoder_hidden_states,
            past_key_value=(past_key, past_value),
            use_cache=True,
        )
        assert len(actual) == 2
        assert_expected(
            actual[1][0],
            torch.cat([past_key, expected_current_key_value], dim=2),
        )
        assert_expected(
            actual[1][1],
            torch.cat([past_value, expected_current_key_value], dim=2),
        )

    @pytest.mark.parametrize(
        "norm_first",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "use_cross_attention",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "use_cache",
        [
            True,
            False,
        ],
    )
    def test_scripting(
        self,
        norm_first,
        use_cross_attention,
        use_cache,
        hidden_states,
        encoder_hidden_states,
        get_decoder_layer,
    ):
        model = get_decoder_layer(
            d_model=4, norm_first=norm_first, use_cross_attention=use_cross_attention
        )
        scripted_model = torch.jit.script(model)
        assert_expected(
            scripted_model(hidden_states, encoder_hidden_states, use_cache=use_cache),
            model(hidden_states, encoder_hidden_states, use_cache=use_cache),
            rtol=0,
            atol=1e-4,
        )


class TestTransformerDecoder:
    @pytest.fixture
    def get_decoder(self):
        def create_decoder(
            d_model,
            norm_first,
            use_cross_attention,
            final_layer_norm_eps=None,
            custom_init=False,
        ):
            model = TransformerDecoder(
                n_layer=2,
                d_model=d_model,
                n_head=1,
                dim_feedforward=2,
                norm_first=norm_first,
                final_layer_norm_eps=final_layer_norm_eps,
                use_cross_attention=use_cross_attention,
            )
            init_weights_with_constant(model)
            if custom_init:
                with torch.no_grad():
                    # Manually override layernorm params to give unique results
                    for i, layer in enumerate(model.layer):
                        for name, param in layer.named_parameters():
                            if "norm" in name:
                                param.copy_(
                                    i
                                    * torch.arange(param.shape[0]).to(dtype=torch.float)
                                )

            model.eval()
            return model

        return create_decoder

    @pytest.fixture
    def inputs(self):
        return Tensor([[[2, 3], [1, 2]]])

    @pytest.fixture
    def hidden_states(self):
        return Tensor([[[1, 2, 3, 4], [4, 2, 0, 2]]])

    @pytest.fixture
    def encoder_hidden_states(self):
        return Tensor([[[5, 6, 7, 8], [8, 9, 11, 12], [2, 1, 0, 2], [0, 0, 4, 4]]])

    @pytest.mark.parametrize(
        "norm_first, return_hidden_states, expected_output",
        [
            (
                True,
                False,
                TransformerOutput(
                    last_hidden_state=Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    hidden_states=[],
                    current_key_values=[],
                ),
            ),
            (
                False,
                False,
                TransformerOutput(
                    last_hidden_state=Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    hidden_states=[],
                    current_key_values=[],
                ),
            ),
            (
                True,
                True,
                TransformerOutput(
                    last_hidden_state=Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    hidden_states=[
                        Tensor([[[2.0, 3.0], [1.0, 2.0]]]),
                        Tensor([[[16.0, 17.0], [15.0, 16.0]]]),
                        Tensor([[[30.0, 31.0], [29.0, 30.0]]]),
                    ],
                    current_key_values=[],
                ),
            ),
            (
                False,
                True,
                TransformerOutput(
                    last_hidden_state=Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    hidden_states=[
                        Tensor([[[2.0, 3.0], [1.0, 2.0]]]),
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                        Tensor([[[0.0, 2.0], [0.0, 2.0]]]),
                    ],
                    current_key_values=[],
                ),
            ),
        ],
    )
    def test_forward_without_cross_attention(
        self, inputs, norm_first, return_hidden_states, expected_output, get_decoder
    ):
        model = get_decoder(d_model=2, norm_first=norm_first, use_cross_attention=False)
        actual_output = model(inputs, return_hidden_states=return_hidden_states)
        # strict keyword breaks in python 3.8/3.9 so breaks CI tests, hence the noqa
        for actual, expected in zip(actual_output, expected_output):  # noqa
            assert_expected(actual, expected, rtol=0, atol=1e-4)

    @pytest.mark.parametrize(
        "norm_first, expected_output",
        [
            (
                True,
                Tensor(
                    [
                        [
                            [409.8329, 410.8329, 411.8329, 412.8329],
                            [398.2318, 396.2318, 394.2318, 396.2318],
                        ]
                    ]
                ),
            ),
            (
                False,
                Tensor(
                    [
                        [
                            [0.0000, 0.2535, 2.1998, 7.7787],
                            [0.0000, 0.2535, 2.1998, 7.7787],
                        ]
                    ]
                ),
            ),
        ],
    )
    def test_forward_with_cross_attention(
        self,
        hidden_states,
        encoder_hidden_states,
        norm_first,
        expected_output,
        get_decoder,
    ):
        model = get_decoder(
            d_model=4, norm_first=norm_first, use_cross_attention=True, custom_init=True
        )
        actual = model(
            hidden_states, encoder_hidden_states, return_hidden_states=False
        )[0]
        assert_expected(actual, expected_output, rtol=0, atol=1e-3)

    @pytest.mark.parametrize(
        "norm_first",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "use_cross_attention",
        [
            True,
            False,
        ],
    )
    @pytest.mark.parametrize(
        "use_cache",
        [
            True,
            False,
        ],
    )
    def test_scripting(
        self,
        norm_first,
        use_cross_attention,
        use_cache,
        hidden_states,
        encoder_hidden_states,
        get_decoder,
    ):
        model = get_decoder(
            d_model=4, norm_first=norm_first, use_cross_attention=use_cross_attention
        )
        scripted_model = torch.jit.script(model)
        assert_expected(
            scripted_model(hidden_states, encoder_hidden_states, use_cache=use_cache),
            model(hidden_states, encoder_hidden_states, use_cache=use_cache),
            rtol=0,
            atol=1e-4,
        )
