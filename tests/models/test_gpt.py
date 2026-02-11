# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, assert_expected_namedtuple, set_rng_seed
from torch import nn
from torch.nn import functional as F
from torchmultimodal.models.video_gpt.gpt import (
    MultimodalGPT,
    MultimodalGPTOutput,
    MultimodalTransformerDecoder,
    RightShift,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderOutput,
    TransformerLayerOutput,
)
from torchmultimodal.utils.common import shift_dim


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
def num_emb():
    return 6


@pytest.fixture
def latent_shape():
    # the product of dims should equal out_seq_len
    return (1, 1, 4)


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
def num_in_tokens():
    return 4


@pytest.fixture
def num_out_tokens():
    return 6


@pytest.fixture
def in_tokens(in_seq_len):
    return torch.arange(in_seq_len).unsqueeze(0)  # (b, seq_len)


@pytest.fixture
def out_tokens(out_seq_len):
    return torch.arange(out_seq_len).unsqueeze(0)  # (b, seq_len)


@pytest.fixture
def in_modality(in_seq_len, d_model):
    return torch.rand(1, in_seq_len, d_model)  # (b, seq_len, d_model)


@pytest.fixture
def out_modality(out_seq_len, d_model):
    return torch.rand(1, out_seq_len, d_model)  # (b, seq_len, d_model)


@pytest.fixture
def decoder_input(d_model):
    return torch.rand(1, 3, d_model)  # (b, seq_len, d_model)


@pytest.fixture
def attn_mask():
    def _attn_mask(q_seq_len, k_seq_len=None):
        if k_seq_len is None:
            k_seq_len = q_seq_len
        return torch.tril(torch.ones(q_seq_len, k_seq_len))  # (q_seq_len, k_seq_len)

    return _attn_mask


@pytest.fixture
def logits_mask(in_seq_len, out_seq_len, num_in_tokens, num_out_tokens):
    total_seq_len = in_seq_len + out_seq_len
    num_tokens = num_in_tokens + num_out_tokens
    logits_mask = torch.ones(total_seq_len, num_tokens)
    logits_mask[in_seq_len:, :num_in_tokens] = 0
    logits_mask[:in_seq_len, num_in_tokens:] = 0

    return logits_mask


@pytest.fixture
def num_layers():
    return 1


@pytest.fixture
def right_shift(d_model):
    return RightShift(d_model)


@pytest.fixture
def in_projection(d_model, emb_dim):
    return nn.Linear(emb_dim, d_model)


@pytest.fixture
def out_projection(d_model, emb_dim):
    return nn.Linear(emb_dim, d_model)


@pytest.fixture
def in_pos_emb(in_seq_len, d_model):
    return nn.Embedding(in_seq_len, d_model)


@pytest.fixture
def out_pos_emb(out_seq_len, d_model):
    return nn.Embedding(out_seq_len, d_model)


@pytest.fixture
def decoder_layer(n_head, d_model):
    return TransformerDecoderLayer(d_model, n_head=n_head).eval()


@pytest.fixture
def decoder(decoder_layer, num_layers):
    return TransformerDecoder(decoder_layer, num_layers)


@pytest.fixture
def mm_decoder(in_pos_emb, out_pos_emb, decoder, right_shift):
    return MultimodalTransformerDecoder(in_pos_emb, out_pos_emb, decoder, right_shift)


@pytest.fixture
def tokenizer(num_emb, emb_dim):
    class DummyTokenizer(nn.Module):
        def __init__(self, num_emb, emb_dim):
            super().__init__()
            # encoder and decoder do not enter either the training or the token
            # generation paths so we do not test their actual logic but only
            # the interfaces.
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            self.embedding = nn.Parameter(
                torch.arange(num_emb * emb_dim, dtype=torch.float).reshape(
                    num_emb, emb_dim
                )
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, token_ids):
            return self.decoder(shift_dim(self.lookup(token_ids), -1, 1))

        def lookup(self, token_ids):
            return F.embedding(token_ids, self.embedding)

    return DummyTokenizer(num_emb, emb_dim)


@pytest.fixture
def gpt(
    d_model,
    num_in_tokens,
    num_out_tokens,
    mm_decoder,
    latent_shape,
    tokenizer,
    in_projection,
    out_projection,
):
    def _gpt(in_tokenizer=tokenizer, out_tokenizer=tokenizer, use_gpt_init=False):
        return MultimodalGPT(
            d_model=d_model,
            num_in_tokens=num_in_tokens,
            num_out_tokens=num_out_tokens,
            latent_shape=latent_shape,
            in_tokenizer=in_tokenizer,
            out_tokenizer=out_tokenizer,
            mm_decoder=mm_decoder,
            in_projection=in_projection,
            out_projection=out_projection,
            norm_layer=None,
            use_gpt_init=use_gpt_init,
        ).eval()

    return _gpt


def get_pos_ids(x):
    b, seq_len, _ = x.shape
    pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
    return pos_ids[None, :]  # (1, seq_len)


class TestMultimodalGPT:
    def test_tokenizers_missing_methods(self, gpt):
        class BadTokenizer(nn.Module):
            def __init__(self):
                super().__init__()

        with pytest.raises(AttributeError):
            gpt(in_tokenizer=BadTokenizer())

        with pytest.raises(AttributeError):
            gpt(out_tokenizer=BadTokenizer())

    def test_initialize_parameters(self, gpt, mocker):
        # Testing mean and std of the initialized weights data requires a large
        # amount samples to be statistically stable. Here we just test whether
        # the method in question has been called to avoid test flakiness.
        mock_init = mocker.patch("torchmultimodal.models.video_gpt.gpt.Tensor.normal_")
        gpt = gpt(use_gpt_init=True)
        mock_init.assert_called()

    def test_encode_invalid_modality(self, gpt):
        gpt = gpt()
        with pytest.raises(ValueError):
            gpt.encode(torch.randn(1, 2, 3), modality="abc")

    def test_decode_tokens_wrong_shape(self, gpt):
        bad_out_tokens = torch.arange(3)  # seq_len no batch dim
        gpt = gpt()
        with pytest.raises(ValueError):
            gpt.decode(bad_out_tokens)

    def test_decode_tokens_reshape(self, gpt, out_tokens):
        gpt = gpt()
        actual = gpt.decode(out_tokens)
        expected_shape = torch.Size([1, 5, 1, 1, 4])  # (b, emb_dim, *latent_shape)
        assert_expected(actual.shape, expected_shape)

    def test_lookup_invalid_modality(self, gpt):
        gpt = gpt()
        token_ids = torch.arange(3).unsqueeze(0)
        with pytest.raises(ValueError):
            gpt.lookup(token_ids, modality="abc")

    def test_lookup_in_modality(self, gpt, in_tokens):
        gpt = gpt()
        actual = gpt.lookup(in_tokens, "in")
        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                ],
            ]
        )
        assert_expected(actual, expected)

    def test_lookup_out_modality(self, gpt, out_tokens):
        gpt = gpt()
        actual = gpt.lookup(out_tokens, "out")
        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                ],
            ]
        )
        assert_expected(actual, expected)

    def test_fwd_bad_input(self, gpt):
        gpt = gpt()
        with pytest.raises(ValueError):
            gpt.fwd()

    def test_fwd_for_generation(self, gpt, in_tokens, d_model, n_head, mocker):
        """Test autoregressive decoding for one step"""
        gpt = gpt()

        mock_right_shift = mocker.patch.object(
            gpt.mm_decoder.right_shift,
            "forward",
            wraps=gpt.mm_decoder.right_shift.forward,
        )

        b, in_seq_len = in_tokens.shape
        # learn the key/value representation from the full input modality sequence
        with torch.no_grad():
            actual = gpt.fwd(
                in_tokens=in_tokens, use_cache=True, causal=True, right_shift=True
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
        # right shift should be switched on to prepend SOS to input modality sequence
        mock_right_shift.assert_called_once()
        mock_right_shift.reset_mock()

        # generate out_modality for one step
        # take the last in_modality token as the starting token for out_modality generation
        decode_step = 0
        with torch.no_grad():
            actual = gpt.fwd(
                out_tokens=in_tokens[:, -1:],
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
        # right shift should be switched off as the "SOS" token for output modality
        # is the last token of in_modality
        mock_right_shift.assert_not_called()

    def test_forward(
        self,
        gpt,
        in_tokens,
        out_tokens,
        attn_mask,
    ):
        gpt = gpt()

        b, in_seq_len = in_tokens.shape
        b, out_seq_len = out_tokens.shape
        attn_mask = attn_mask(in_seq_len + out_seq_len)
        actual = gpt(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            attn_mask=attn_mask,
            use_cache=True,
            causal=True,
            right_shift=True,
        )
        assert isinstance(actual, MultimodalGPTOutput)
        expected = {
            "decoder_output": {
                "last_hidden_states": (
                    torch.Size([1, 7, 4]),  # (b, seq_len, d_model)
                    64.3909,
                ),
                "hidden_states": None,
                "past_key_values": (
                    (
                        {
                            "k": (torch.Size([1, 2, 7, 2]), 8.3626),
                            "v": (torch.Size([1, 2, 7, 2]), 3.4256),
                        }
                    ),
                ),  # (num_layers, key/value, (b, n_head, seq_len, d_model // n_head)
            },
            "logits": (torch.Size([1, 7, 10]), 0.0),  # (b, seq_len, tokens)
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_logits_mask(
        self,
        gpt,
        in_tokens,
        out_tokens,
        attn_mask,
        logits_mask,
    ):
        gpt = gpt()

        b, in_seq_len = in_tokens.shape
        b, out_seq_len = out_tokens.shape
        attn_mask = attn_mask(in_seq_len + out_seq_len)
        out = gpt(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            attn_mask=attn_mask,
            use_cache=True,
            causal=True,
            right_shift=True,
            logits_mask=logits_mask,
        )
        assert isinstance(out, MultimodalGPTOutput)
        actual = out.logits  # (b, seq_len, num_tokens)
        max_neg_value = -torch.finfo(torch.float32).max
        # assert each quandrant of the logits matrix (b, total_seq_len, num_total_tokens)
        assert_expected(
            actual[:, :3, :4], torch.zeros(1, 3, 4)
        )  # (b, in_seq_len, num_in_tokens)
        assert_expected(
            actual[:, :3, 4:], torch.ones(1, 3, 6) * max_neg_value
        )  # (b, in_seq_len, num_out_tokens)
        assert_expected(
            actual[:, 3:, :4], torch.ones(1, 4, 4) * max_neg_value
        )  # (b, out_seq_len, num_in_tokens)
        assert_expected(
            actual[:, 3:, 4:], torch.zeros(1, 4, 6)
        )  # (b, out_seq_len, num_out_tokens)


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
                0.2222,
            ),  # (b, in_seq_len, d_model)
            "hidden_states": None,
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_out_modality(self, mm_decoder, out_modality):
        actual = mm_decoder(
            out_modality=out_modality, out_pos_ids=get_pos_ids(out_modality)
        )
        expected = {
            "last_hidden_states": (
                torch.Size([1, 4, 4]),
                5.2093,
            ),  # (b, out_seq_len, d_model)
            "hidden_states": None,
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_two_modality(self, mm_decoder, in_modality, out_modality):
        actual = mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=get_pos_ids(in_modality),
            out_pos_ids=get_pos_ids(out_modality),
        )
        expected = {
            "last_hidden_states": (
                torch.Size([1, 7, 4]),
                7.9519,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

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
                7.9519,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

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
                10.1681,
            ),  # (b, in_seq_len + out_seq_len, d_model)
            "hidden_states": None,
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

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
    def test_forward_mask_extended(self, decoder, decoder_input, attn_mask, num_layers):
        b, seq_len, _ = decoder_input.shape
        attn_mask = attn_mask(seq_len).unsqueeze(0)  # add batch dim
        actual = decoder(decoder_input, attn_mask)
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))

    def test_forward(self, decoder, decoder_input, attn_mask, num_layers):
        b, seq_len, _ = decoder_input.shape
        attn_mask = attn_mask(seq_len)
        actual = decoder(decoder_input, attn_mask)
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))

    def test_forward_additional_output(self, decoder, decoder_input, num_layers):
        actual = decoder(
            decoder_input,
            use_cache=True,
            return_hidden_states=True,
        )
        assert isinstance(actual, TransformerDecoderOutput)
        assert_expected(actual.last_hidden_states.shape, torch.Size([1, 3, 4]))
        assert_expected(
            len(actual.hidden_states), num_layers + 1
        )  # +1 to include the input hidden_states
        assert_expected(len(actual.past_key_values), num_layers)


class TestTransformerDecoderLayer:
    def test_forward(self, decoder_layer, decoder_input):
        actual = decoder_layer(decoder_input)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 5.1956),  # (b, seq_len, d_model)
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_masked(self, decoder_layer, decoder_input, attn_mask):
        b, seq_len, _ = decoder_input.shape
        attn_mask = attn_mask(seq_len)
        actual = decoder_layer(decoder_input, attn_mask)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 5.6602),  # (b, seq_len, seq_len)
            "past_key_values": None,
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)

    def test_forward_additional_output(self, decoder_layer, decoder_input):
        actual = decoder_layer(decoder_input, use_cache=True)
        assert isinstance(actual, TransformerLayerOutput)
        expected = {
            "hidden_states": (torch.Size([1, 3, 4]), 5.1956),  # (b, seq_len, seq_len)
            "past_key_values": {
                "k": ([1, 2, 3, 2], 4.8075),
                "v": ([1, 2, 3, 2], -5.6613),
            },  # (b, h, seq_len, d_model//h)
        }
        assert_expected_namedtuple(actual, expected, rtol=1e-5, atol=1e-4)


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
