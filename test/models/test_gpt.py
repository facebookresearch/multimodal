# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, assert_expected_wrapper, set_rng_seed
from torch import nn
from torchmultimodal.models.gpt import (
    MultimodalTransformerDecoder,
    RightShift,
    SiLU,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderOutput,
    TransformerLayerOutput,
)


def tuple_to_dict(t):
    if not isinstance(t, tuple):
        raise TypeError(f"Input must be of type tuple but got {type(t)}")

    return {k: v for k, v in enumerate(t)}


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def d_model():
    return 4


@pytest.fixture
def emb_dim():
    return 5


@pytest.fixture
def in_seq_len():
    return 3


@pytest.fixture
def out_seq_len():
    return 4


@pytest.fixture
def n_head():
    return 2


@pytest.fixture
def right_shift(d_model):
    return RightShift(d_model)


@pytest.fixture
def in_token_emb(d_model, emb_dim):
    return nn.Linear(emb_dim, d_model)


@pytest.fixture
def out_token_emb(d_model, emb_dim):
    return nn.Linear(emb_dim, d_model)


@pytest.fixture
def in_pos_emb(in_seq_len, d_model):
    return nn.Embedding(in_seq_len, d_model)


@pytest.fixture
def out_pos_emb(out_seq_len, d_model):
    return nn.Embedding(out_seq_len, d_model)


@pytest.fixture
def in_modality(in_seq_len, emb_dim):
    return torch.rand(1, in_seq_len, emb_dim)  # (b, seq_len, emb_dim)


@pytest.fixture
def out_modality(out_seq_len, emb_dim):
    return torch.rand(1, out_seq_len, emb_dim)  # (b, seq_len, emb_dim)


@pytest.fixture
def x_input(d_model):
    return torch.rand(1, 3, d_model)  # (b, seq_len, emb_dim)


@pytest.fixture
def self_attn_mask(in_seq_len):
    return torch.tril(torch.ones(in_seq_len, in_seq_len))[
        None, :
    ]  # (b, seq_len, seq_len)


@pytest.fixture
def self_head_mask(n_head, in_seq_len):
    masked = torch.zeros(1, in_seq_len, in_seq_len)
    unmasked = torch.ones(n_head - 1, in_seq_len, in_seq_len)
    return torch.cat((masked, unmasked), dim=0)[None, :]  # (b, h, seq_len, seq_len)


@pytest.fixture
def num_layers():
    return 1


@pytest.fixture
def decoder_layer(n_head, d_model):
    return TransformerDecoderLayer(d_model, n_head=n_head).eval()


@pytest.fixture
def decoder(decoder_layer, num_layers):
    return TransformerDecoder(decoder_layer, num_layers)


@pytest.fixture
def mm_decoder(
    in_token_emb, out_token_emb, in_pos_emb, out_pos_emb, decoder, right_shift
):
    return MultimodalTransformerDecoder(
        in_token_emb, out_token_emb, in_pos_emb, out_pos_emb, decoder, right_shift
    )


class TestMultimodalTransformerDecoder:
    def _pos_ids(self, x):
        b, seq_len, _ = x.shape
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return pos_ids[None, :]  # (b, seq_len)

    def test_bad_input(self, mm_decoder):
        with pytest.raises(ValueError):
            mm_decoder()

    def test_forward_in_modality(self, mm_decoder, in_modality):
        actual = mm_decoder(
            in_modality=in_modality, in_pos_ids=self._pos_ids(in_modality)
        )
        expected = {
            "last_hidden_states": (
                torch.Size([1, 3, 4]),
                -0.5538,
            ),  # (b, in_seq_len, d_model)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_out_modality(self, mm_decoder, out_modality):
        actual = mm_decoder(
            out_modality=out_modality, out_pos_ids=self._pos_ids(out_modality)
        )
        expected = {
            "last_hidden_states": (
                torch.Size([1, 4, 4]),
                -0.3621,
            ),  # (b, out_seq_len, d_model)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_two_modality(self, mm_decoder, in_modality, out_modality):
        actual = mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=self._pos_ids(in_modality),
            out_pos_ids=self._pos_ids(out_modality),
        )
        print(actual.last_hidden_states.sum())
        expected = {
            "last_hidden_states": (
                torch.Size([1, 7, 4]),
                -0.8250,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_eval_right_shift_on(
        self, mm_decoder, in_modality, out_modality, mocker
    ):
        """Test right shift is switched on during eval mode"""
        mock_right_shift = mocker.patch.object(
            mm_decoder.right_shift, "forward", wraps=mm_decoder.right_shift.forward
        )

        mm_decoder.eval()
        actual = mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=self._pos_ids(in_modality),
            out_pos_ids=self._pos_ids(out_modality),
            right_shift=True,
        )
        mock_right_shift.assert_called_once()
        expected = {
            "last_hidden_states": (
                torch.Size([1, 7, 4]),
                -0.8250,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_eval_right_shift_off(
        self, mm_decoder, in_modality, out_modality, mocker
    ):
        """Test right shift is switched off during eval mode"""
        mock_right_shift = mocker.patch.object(
            mm_decoder.right_shift, "forward", wraps=mm_decoder.right_shift.forward
        )

        mm_decoder.eval()
        actual = mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=self._pos_ids(in_modality),
            out_pos_ids=self._pos_ids(out_modality),
        )
        mock_right_shift.assert_not_called()
        expected = {
            "last_hidden_states": (
                torch.Size([1, 7, 4]),
                -1.0925,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_bad_pos_ids(self, mm_decoder, in_modality, in_seq_len):
        in_pos_ids = torch.arange(
            in_seq_len + 1, dtype=torch.long, device=in_modality.device
        )[None, :]
        with pytest.raises(ValueError):
            mm_decoder._norm_pos_ids(in_modality, in_pos_ids)

    def test_optional_pos_ids(self, mm_decoder, in_modality):
        actual = mm_decoder._norm_pos_ids(in_modality)
        expected = self._pos_ids(in_modality)
        assert_expected(actual, expected)


class TestTransformerDecoder:
    def test_forward(
        self, decoder, x_input, self_attn_mask, self_head_mask, num_layers
    ):
        actual = decoder(x_input, self_attn_mask, self_head_mask)
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))

    def test_forward_additional_output(self, decoder, x_input, num_layers):
        actual = decoder(
            x_input,
            use_cache=True,
            return_attn_weights=True,
            return_hidden_states=True,
        )
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))
        assert_expected(
            len(actual.hidden_states), num_layers + 1
        )  # +1 to include the input hidden_states
        assert_expected(len(actual.attention_weights), num_layers)
        assert_expected(len(actual.past_key_values), num_layers)


class TestTransformerDecoderLayer:
    def test_forward(self, decoder_layer, x_input):
        actual = decoder_layer(x_input)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 0.4808),  # (b, seq_len, d_model)
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_masked(
        self, decoder_layer, x_input, self_attn_mask, self_head_mask
    ):
        actual = decoder_layer(x_input, self_attn_mask, self_head_mask)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 1.5776),  # (b, seq_len, seq_len)
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_additional_output(self, decoder_layer, x_input):
        actual = decoder_layer(x_input, use_cache=True, return_attn_weights=True)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 0.4808),  # (b, seq_len, seq_len)
            "attention_weights": (
                torch.Size([1, 2, 3, 3]),
                6.0,
            ),  # (b, h, seq_len, seq_len)
            "past_key_values": {
                "k": ([1, 2, 3, 2], -0.3156),
                "v": ([1, 2, 3, 2], 5.1630),
            },  # (b, h, seq_len, d_model//h)
        }
        assert_expected_wrapper(actual, expected, rtol=1e-5, atol=1e-4)


def test_sigmoid_linear_unit():
    silu = SiLU()
    actual = silu(torch.ones(3))
    expected = torch.tensor([0.8458, 0.8458, 0.8458])
    assert_expected(actual, expected)


def test_right_shift(right_shift, d_model):
    x = torch.ones(1, 3, d_model)  # (b, seq_len, d_model)
    actual = right_shift(x)
    expected = torch.tensor(
        [
            [
                [-0.0321, 0.0046, 0.0448, 0.0169],
                [1.0000, 1.0000, 1.0000, 1.0000],
                [1.0000, 1.0000, 1.0000, 1.0000],
            ]
        ]
    )
    assert_expected(actual, expected, rtol=1e-5, atol=1e-4)
