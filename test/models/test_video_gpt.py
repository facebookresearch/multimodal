# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected, set_rng_seed

from torchmultimodal.models.video_gpt import video_gpt


@pytest.fixture(autouse=True)
def set_seed():
    return set_rng_seed(4)


_model_params = {
    "input_shape": (16, 64, 64),
    "latent_shape": (8, 32, 32),
    "d_model": 576,
    "n_head": 4,
    "dropout": 0.2,
    "attn_dropout": 0.3,
    "num_decoder_layers": 16,
}


@pytest.fixture
def model():
    return video_gpt(**_model_params).eval()


def test_encode(model):
    x = torch.randn((1, 3, *_model_params["input_shape"]))  # (b, c, *input_shape)
    actual = model.encode(x, "in")
    assert_expected(actual.shape, (1, 8192))
    assert_expected(actual.sum().item(), 6678698)


def test_decode(model):
    latent_seq_len = torch.prod(torch.tensor(_model_params["latent_shape"])).item()
    x = torch.randint(0, 10, (1, latent_seq_len))  # tokens
    actual = model.decode(x)
    assert_expected(actual.shape, (1, 3, 16, 64, 64))
    assert_expected(actual.sum().item(), 14651.1406, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "modality, expected_shape, expected_sum",
    [("in", (1, 2, 256), 38.8214), ("out", (1, 2, 256), -23.4659)],
)
def test_lookup(model, modality, expected_shape, expected_sum):
    x = torch.tensor([[1, 2]])  # tokens
    actual = model.lookup(x, modality)
    assert_expected(actual.shape, expected_shape)
    assert_expected(actual.sum().item(), expected_sum, rtol=1e-5, atol=1e-4)


def test_forward(model):
    n_head = _model_params["n_head"]

    x = torch.tensor([[1, 2, 3, 4]])  # (b, in_seq_len)
    y = torch.tensor([[5, 6, 7]])  # (b, out_seq_len)
    attn_mask = torch.tril(torch.ones(7, 7)).unsqueeze(0)  # (b, seq_len, seq_len)
    head_mask = torch.ones(1, n_head, 7, 7)  # (b, h, seq_len, seq_len)

    num_tokens = model.num_in_tokens + model.num_out_tokens
    logits_mask = torch.ones(1, 7, num_tokens)  # (b, seq_len, num_tokens)

    out = model(
        x,
        y,
        attn_mask=attn_mask,
        head_mask=head_mask,
        logits_mask=logits_mask,
        use_cache=True,
        causal=True,
        return_attn_weights=True,
        return_hidden_states=True,
    )
    actual = out.decoder_output.last_hidden_states
    assert_expected(actual.shape, (1, 7, 576))
    assert_expected(actual.sum().item(), 0.7185, rtol=1e-5, atol=1e-4)
