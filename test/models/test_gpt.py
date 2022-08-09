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
    MultimodalGPT,
    MultimodalGPTOutput,
    MultimodalTransformerDecoder,
    RightShift,
    SiLU,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderOutput,
    TransformerLayerOutput,
)


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
def num_in_tokens():
    return 4


@pytest.fixture
def num_out_tokens():
    return 6


@pytest.fixture
def num_tokens(num_in_tokens, num_out_tokens):
    return num_in_tokens + num_out_tokens


@pytest.fixture
def self_attn_mask():
    def _attn_mask(seq_len):
        return torch.tril(torch.ones(seq_len, seq_len))[
            None, :
        ]  # (b, seq_len, seq_len)

    return _attn_mask


@pytest.fixture
def self_head_mask(n_head):
    def _head_mask(seq_len):
        masked = torch.zeros(1, seq_len, seq_len)
        unmasked = torch.ones(n_head - 1, seq_len, seq_len)
        return torch.cat((masked, unmasked), dim=0)[None, :]  # (b, h, seq_len, seq_len)

    return _head_mask


@pytest.fixture
def logits_mask(in_seq_len, out_seq_len, num_in_tokens, num_out_tokens):
    total_seq_len = in_seq_len + out_seq_len
    num_tokens = num_in_tokens + num_out_tokens
    logits_mask = torch.ones(1, total_seq_len, num_tokens)
    logits_mask[:, in_seq_len:, :num_in_tokens] = 0
    logits_mask[:, :in_seq_len, num_in_tokens:] = 0

    return logits_mask


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


@pytest.fixture
def gpt(d_model, num_tokens, mm_decoder):
    class DummyTokenizer(nn.Module):
        def __init__(self):
            super().__init__()

        def encode(self, x):
            pass

        def decode(self, token_ids):
            pass

        def lookup(self, token_ids):
            pass

    def _gpt(in_tokenizer=DummyTokenizer(), out_tokenizer=DummyTokenizer()):
        return MultimodalGPT(
            d_model=d_model,
            num_tokens=num_tokens,
            in_tokenizer=in_tokenizer,
            out_tokenizer=out_tokenizer,
            mm_decoder=mm_decoder,
        ).eval()

    return _gpt


def get_pos_ids(x):
    b, seq_len, _ = x.shape
    pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
    return pos_ids[None, :]  # (b, seq_len)


class TestMultimodalGPT:
    def test_bad_tokenizers(self, gpt):
        class BadTokenizer(nn.Module):
            def __init__(self):
                super().__init__()

        with pytest.raises(AttributeError):
            gpt(in_tokenizer=BadTokenizer())

        with pytest.raises(AttributeError):
            gpt(out_tokenizer=BadTokenizer())

    def test_fwd_for_generation(self, gpt, in_modality, d_model, n_head, mocker):
        """Test autoregressive decoding for one step"""
        gpt = gpt()

        mock_right_shift = mocker.patch.object(
            gpt.mm_decoder.right_shift,
            "forward",
            wraps=gpt.mm_decoder.right_shift.forward,
        )

        b, in_seq_len, _ = in_modality.shape
        # learn the key/value representation from the full in_modality sequence
        with torch.no_grad():
            actual = gpt.fwd(
                in_modality=in_modality, use_cache=True, causal=True, right_shift=True
            )
        assert isinstance(actual, TransformerDecoderOutput)
        # check that the key/value representation has been learnt
        for layer_past_kv in actual.past_key_values:
            assert_expected(
                layer_past_kv["k"].shape,
                torch.Size([1, 2, 3, 2]),  # (b, n_head, in_seq_len, d_model // n_head)
            )
            assert_expected(
                layer_past_kv["v"].shape,
                torch.Size([1, 2, 3, 2]),
            )
        # right shift should be switched on to prepend SOS to in_modality sequence
        mock_right_shift.assert_called_once()
        mock_right_shift.reset_mock()

        # generate out_modality for one step
        # take the last in_modality token as the starting token for out_modality generation
        decode_step = 0
        with torch.no_grad():
            actual = gpt.fwd(
                out_modality=in_modality[:, -1:, :],
                out_pos_ids=torch.tensor([decode_step]).unsqueeze(0).repeat(b, 1),
                use_cache=True,
                causal=True,
            )
        # check that the key/value representation has increased by 1 unit
        for layer_past_kv in actual.past_key_values:
            assert_expected(
                layer_past_kv["k"].shape,
                torch.Size(
                    [1, 2, 4, 2]
                ),  # (b, n_head, in_seq_len + 1, d_model // n_head)
            )
            assert_expected(
                layer_past_kv["v"].shape,
                torch.Size([1, 2, 4, 2]),
            )
        # right shift should be switched off as the "SOS" token for out_modality is the last token
        # of in_modality
        mock_right_shift.assert_not_called()

    def test_forward(
        self,
        gpt,
        in_modality,
        out_modality,
        self_attn_mask,
        self_head_mask,
        logits_mask,
    ):
        gpt = gpt()

        b, in_seq_len, _ = in_modality.shape
        b, out_seq_len, _ = out_modality.shape
        attn_mask = self_attn_mask(in_seq_len + out_seq_len)
        head_mask = self_head_mask(in_seq_len + out_seq_len)
        actual = gpt.forward(
            in_modality=in_modality,
            out_modality=out_modality,
            attn_mask=attn_mask,
            head_mask=head_mask,
            use_cache=True,
            causal=True,
            right_shift=True,
            logits_mask=logits_mask,
        )
        assert isinstance(actual, MultimodalGPTOutput)
        expected = {
            "decoder_output": {
                "last_hidden_states": (
                    torch.Size([1, 7, 4]),  # (b, seq_len, d_model)
                    -0.2385,
                ),
                "hidden_states": None,
                "attention_weights": None,
                "past_key_values": (
                    (
                        {
                            "k": (torch.Size([1, 2, 7, 2]), 6.0892),
                            "v": (torch.Size([1, 2, 7, 2]), 2.6072),
                        }
                    ),
                ),  # (num_layers, key/value, (b, n_head, seq_len, d_model // n_head)
            },
            "logits": (torch.Size([1, 10, 7]), 0.0),  # (b, tokens, seq_len)
        }


class TestMultimodalTransformerDecoder:
    def test_bad_input(self, mm_decoder):
        with pytest.raises(ValueError):
            mm_decoder()

    def test_forward_in_modality(self, mm_decoder, in_modality):
        actual = mm_decoder(
            in_modality=in_modality, in_pos_ids=get_pos_ids(in_modality)
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
        assert_expected_wrapper(actual, expected)

    def test_forward_out_modality(self, mm_decoder, out_modality):
        actual = mm_decoder(
            out_modality=out_modality, out_pos_ids=get_pos_ids(out_modality)
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
        assert_expected_wrapper(actual, expected)

    def test_forward_two_modality(self, mm_decoder, in_modality, out_modality):
        actual = mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=get_pos_ids(in_modality),
            out_pos_ids=get_pos_ids(out_modality),
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
        assert_expected_wrapper(actual, expected)

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
            in_pos_ids=get_pos_ids(in_modality),
            out_pos_ids=get_pos_ids(out_modality),
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
        assert_expected_wrapper(actual, expected)

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
            in_pos_ids=get_pos_ids(in_modality),
            out_pos_ids=get_pos_ids(out_modality),
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
        assert_expected_wrapper(actual, expected)

    def test_bad_pos_ids(self, mm_decoder, in_modality, in_seq_len):
        in_pos_ids = torch.arange(
            in_seq_len + 1, dtype=torch.long, device=in_modality.device
        )[None, :]
        with pytest.raises(ValueError):
            mm_decoder._norm_pos_ids(in_modality, in_pos_ids)

    def test_optional_pos_ids(self, mm_decoder, in_modality):
        actual = mm_decoder._norm_pos_ids(in_modality)
        expected = get_pos_ids(in_modality)
        assert_expected(actual, expected)


class TestTransformerDecoder:
    def test_forward(
        self, decoder, x_input, self_attn_mask, self_head_mask, num_layers
    ):
        b, seq_len, _ = x_input.shape
        attn_mask = self_attn_mask(seq_len)
        head_mask = self_head_mask(seq_len)
        actual = decoder(x_input, attn_mask, head_mask)
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
        assert_expected_wrapper(actual, expected)

    def test_forward_masked(
        self, decoder_layer, x_input, self_attn_mask, self_head_mask
    ):
        b, seq_len, _ = x_input.shape
        attn_mask = self_attn_mask(seq_len)
        head_mask = self_head_mask(seq_len)
        actual = decoder_layer(x_input, attn_mask, head_mask)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 1.5776),  # (b, seq_len, seq_len)
            "attention_weights": None,
            "past_key_values": None,
        }
        assert_expected_wrapper(actual, expected)

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
        assert_expected_wrapper(actual, expected)


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
