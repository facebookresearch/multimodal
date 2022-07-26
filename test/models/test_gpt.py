# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.models.gpt import (
    MultimodalGPT,
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


def assert_expected_wrapper(actual, expected):
    """Helper function that calls assert_expected recursively on nested Dict/NamedTuple"""
    # convert NamedTuple to dictionary
    if isinstance(actual, tuple) and hasattr(
        actual, "_asdict"
    ):  # hacky way to assert an instance of NamedTuple
        actual = actual._asdict()

    if not isinstance(actual, Dict):
        raise TypeError(f"actual needs to be a dictionary but got {type(actual)}")

    if not isinstance(expected, Dict):
        raise TypeError(f"expected needs to be a dictionary but got {type(expected)}")

    for attr, _expected in expected.items():
        _actual = actual[attr]

        if _actual is None:
            # optional output
            assert _expected is None
        elif isinstance(_actual, Dict):
            # dictionary output, e.g., cache of k/v
            assert_expected_wrapper(_actual, _expected)
        elif isinstance(_actual, tuple):
            # outputs are from multiple layers: (Tensor, Tensor, ...)
            assert_expected_wrapper(tuple_to_dict(_actual), tuple_to_dict(_expected))
        elif isinstance(_actual, Tensor):
            # single tensor output
            _expected_shape, _expected_sum = _expected
            assert_expected(_actual.shape, _expected_shape)
            assert_expected(_actual.sum().item(), _expected_sum, rtol=1e-5, atol=1e-4)
        else:
            raise TypeError(
                f"Unsupported types for test assertion: {_actual}, {_expected}"
            )


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def emb_dim():
    return 4


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
def right_shift(emb_dim):
    return RightShift(emb_dim)


@pytest.fixture
def in_modality(in_seq_len, emb_dim):
    return torch.rand(1, in_seq_len, emb_dim)  # (bs, seq_len, emb_dim)


@pytest.fixture
def out_modality(out_seq_len, emb_dim):
    return torch.rand(1, out_seq_len, emb_dim)  # (bs, seq_len, emb_dim)


@pytest.fixture
def self_attn_mask(in_seq_len):
    return torch.tril(torch.ones(in_seq_len, in_seq_len))[
        None, :
    ]  # (bs, seq_len, seq_len)


@pytest.fixture
def self_head_mask(n_head, in_seq_len):
    masked = torch.zeros(1, in_seq_len, in_seq_len)
    unmasked = torch.ones(n_head - 1, in_seq_len, in_seq_len)
    return torch.cat((masked, unmasked), dim=0)[None, :]  # (bs, h, seq_len, seq_len)


@pytest.fixture
def num_layers():
    return 1


@pytest.fixture
def decoder_layer(n_head, emb_dim):
    return TransformerDecoderLayer(d_model=emb_dim, n_head=n_head).eval()


@pytest.fixture
def decoder(decoder_layer, num_layers):
    return TransformerDecoder(decoder_layer, num_layers)


@pytest.fixture
def gpt(decoder, in_seq_len, out_seq_len, emb_dim):
    class DummyTokenEmbedding(nn.Module):
        def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
            _, seq_len, emb_dim = x.shape
            x_tok = torch.arange(seq_len, dtype=torch.long, device=x.device)
            x_emb = x
            return x_tok, x_emb

    return MultimodalGPT(
        in_token_emb=DummyTokenEmbedding(),
        out_token_emb=DummyTokenEmbedding(),
        in_pos_emb=nn.Embedding(in_seq_len, emb_dim),
        out_pos_emb=nn.Embedding(out_seq_len, emb_dim),
        decoder=decoder,
    )


class TestMultimodalGPT:
    def _pos_ids(self, x):
        bs, seq_len, _ = x.shape
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return pos_ids[None, :]  # (bs, seq_len)

    def test_bad_input(self, gpt):
        with pytest.raises(ValueError):
            gpt()

    def test_forward_in_modality(self, gpt, in_modality):
        actual = gpt(in_modality=in_modality, in_pos_ids=self._pos_ids(in_modality))
        expected = {
            "last_hidden_states": (
                torch.Size([1, 3, 4]),
                0.5318,
            ),  # (bs, in_seq_len, emb_dim)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

    def test_forward_out_modality(self, gpt, out_modality):
        actual = gpt(out_modality=out_modality, out_pos_ids=self._pos_ids(out_modality))
        expected = {
            "last_hidden_states": (
                torch.Size([1, 4, 4]),
                0.7538,
            ),  # (bs, out_seq_len, emb_dim)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

    def test_forward_two_modality(self, gpt, in_modality, out_modality):
        actual = gpt(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=self._pos_ids(in_modality),
            out_pos_ids=self._pos_ids(out_modality),
        )
        expected = {
            "last_hidden_states": (
                torch.Size([1, 7, 4]),
                1.7604,
            ),  # (bs, in_seq_len + out_seq_len, emb_dim)
            "hidden_states": None,
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

    def test_encode(self, gpt, in_modality):
        actual = gpt._encode(in_modality, self._pos_ids(in_modality), "in")
        assert_expected(actual.shape, torch.Size([1, 3, 4]))  # (bs, seq_len, emb_dim)
        assert_expected(actual.sum().item(), 0.1456, rtol=1e-5, atol=1e-4)

    def test_bad_pos_ids(self, gpt, in_modality, in_seq_len):
        in_pos_ids = torch.arange(
            in_seq_len + 1, dtype=torch.long, device=in_modality.device
        )[None, :]
        with pytest.raises(ValueError):
            gpt._norm_pos_ids(in_modality, in_pos_ids)

    def test_optional_pos_ids(self, gpt, in_modality):
        actual = gpt._norm_pos_ids(in_modality)
        expected = self._pos_ids(in_modality)
        assert_expected(actual, expected)


class TestTransformerDecoder:
    def test_forward(
        self, decoder, in_modality, self_attn_mask, self_head_mask, num_layers
    ):
        actual = decoder(in_modality, self_attn_mask, self_head_mask)
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))

    def test_forward_additional_output(self, decoder, in_modality, num_layers):
        actual = decoder(
            in_modality,
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
    def test_forward(self, decoder_layer, in_modality):
        actual = decoder_layer(in_modality)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 2.3115),  # (bs, seq_len, emb_dim)
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

    def test_forward_masked(
        self, decoder_layer, in_modality, self_attn_mask, self_head_mask
    ):
        actual = decoder_layer(in_modality, self_attn_mask, self_head_mask)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 1.4288),  # (bs, seq_len, seq_len)
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

    def test_forward_additional_output(self, decoder_layer, in_modality):
        actual = decoder_layer(in_modality, use_cache=True, return_attn_weights=True)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 2.3115),  # (bs, seq_len, seq_len)
            "attention_weights": (
                torch.Size([1, 2, 3, 3]),
                6.0,
            ),  # (bs, h, seq_len, seq_len)
            "past_key_values": {
                "k": ([1, 2, 3, 2], 1.6034),
                "v": ([1, 2, 3, 2], 4.6519),
            },  # (bs, h, seq_len, emb_dim//h)
        }
        assert_expected_wrapper(actual, expected)


def test_sigmoid_linear_unit():
    silu = SiLU()
    actual = silu(torch.ones(3))
    expected = torch.tensor([0.8458, 0.8458, 0.8458])
    assert_expected(actual, expected)


def test_right_shift(right_shift, emb_dim):
    x = torch.ones(1, 3, emb_dim)  # (bs, seq_len, emb_dim)
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
