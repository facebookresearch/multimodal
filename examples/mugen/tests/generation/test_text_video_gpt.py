# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from examples.mugen.generation.text_video_gpt import text_video_gpt

from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.utils.common import get_current_device


@pytest.fixture(autouse=True)
def set_seed():
    return set_rng_seed(0)


@pytest.fixture
def device():
    return get_current_device()


@pytest.fixture
def model_fn():
    return text_video_gpt


_model_params = {
    "text_seq_len": 128,
    "video_seq_len": 32,
    "resolution": 256,
    "downsample": (4, 32, 32),
    "d_model": 768,
    "n_head": 8,
    "dropout": 0.2,
    "attn_dropout": 0.3,
    "num_decoder_layers": 12,
    "use_gpt_init": True,
}


def test_encode_text(model_fn, device):
    test_params = {"text_seq_len": 5}
    kwargs = {**_model_params, **test_params}

    model = model_fn(**kwargs)
    model.eval()
    x = ["MUGEN walks from right to left."]
    actual = model.encode(x, "in", device=device)
    expected = torch.tensor([[80, 118, 110, 85, 70]])
    assert_expected(actual, expected)


@pytest.mark.parametrize(
    "video_seq_len, expected", [(8, (1, 128)), (16, (1, 256)), (32, (1, 512))]
)
def test_encode_video(model_fn, video_seq_len, expected):
    test_params = {"video_seq_len": video_seq_len}
    kwargs = {**_model_params, **test_params}

    video_input_shape = tuple(
        [kwargs[_f] for _f in ["video_seq_len", "resolution", "resolution"]]
    )

    input_shape = (1, 3, *video_input_shape)
    x = torch.rand(input_shape)

    model = model_fn(**kwargs)
    model.eval()
    actual = model.encode(x, "out")
    assert_expected(actual.shape, expected)


@pytest.mark.parametrize(
    "video_seq_len, expected",
    [(8, 55462.1719), (16, 112028.1719), (32, 225157.7656)],
)
def test_decode_video(model_fn, video_seq_len, expected):
    test_params = {"video_seq_len": video_seq_len}
    kwargs = {**_model_params, **test_params}
    model = model_fn(**kwargs)
    model.eval()
    latent_shape = model.latent_shape
    latent_seq_len = torch.prod(torch.tensor(latent_shape)).item()
    x = torch.randint(0, 10, (1, latent_seq_len))  # tokens
    actual = model.decode(x)
    assert_expected(actual.shape, (1, 3, video_seq_len, 256, 256))
    print(actual.sum())
    assert_expected(actual.sum().item(), expected, rtol=1, atol=1e-4)


@pytest.mark.parametrize(
    "video_seq_len, expected",
    [(8, 116013.4766), (16, 237488.6250), (32, 536481.4375)],
)
def test_decode_video_checkpoint(model_fn, video_seq_len, expected):
    vqvae_model_key = f"mugen_L{video_seq_len}"
    test_params = {
        "video_seq_len": video_seq_len,
        "pretrained_video_vqvae_model_key": vqvae_model_key,
    }
    kwargs = {**_model_params, **test_params}
    model = model_fn(**kwargs)
    model.eval()
    latent_shape = model.latent_shape
    latent_seq_len = torch.prod(torch.tensor(latent_shape)).item()
    x = torch.randint(0, 10, (1, latent_seq_len))  # tokens
    actual = model.decode(x)
    assert_expected(actual.shape, (1, 3, video_seq_len, 256, 256))
    assert_expected(actual.sum().item(), expected, rtol=1, atol=1e-4)


@pytest.mark.parametrize(
    "modality, expected_shape, expected_sum",
    [("in", (1, 4, 768), -53.7916), ("out", (1, 4, 256), 42.4742)],
)
def test_lookup(model_fn, modality, expected_shape, expected_sum):
    test_params = {"text_seq_len": 5}
    kwargs = {**_model_params, **test_params}

    x = torch.tensor([[1, 2, 3, 4]])

    model = model_fn(**kwargs)
    model.eval()
    actual = model.lookup(x, modality)
    assert_expected(actual.shape, expected_shape)  # (b, num_tokens, d_model)
    assert_expected(actual.sum().item(), expected_sum, rtol=1, atol=1e-4)


@pytest.mark.parametrize(
    "video_seq_len, expected", [(8, 782.1641), (16, -442.4437), (32, 585.2963)]
)
def test_forward_no_pretrained(model_fn, video_seq_len, expected):
    test_params = {"video_seq_len": video_seq_len}
    kwargs = {**_model_params, **test_params}
    n_head = kwargs["n_head"]

    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[5, 6, 7]])
    attn_mask = torch.tril(torch.ones(7, 7)).unsqueeze(0)  # (b, seq_len, seq_len)
    head_mask = torch.ones(1, n_head, 7, 7)  # (b, h, seq_len, seq_len)

    model = model_fn(**kwargs)
    model.eval()
    num_tokens = model.num_in_tokens + model.num_out_tokens
    logits_mask = torch.ones(1, 7, num_tokens)  # (b, seq_len, num_tokens)

    out = model(x, y, attn_mask=attn_mask, head_mask=head_mask, logits_mask=logits_mask)
    actual = out.decoder_output.last_hidden_states
    assert_expected(actual.shape, (1, 7, 768))
    assert_expected(actual.sum().item(), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "video_seq_len, expected",
    [(8, 431.3439), (16, -180.2783), (32, 462.27)],
)
def test_forward_vqvae_pretrained(model_fn, video_seq_len, expected):
    vqvae_model_key = f"mugen_L{video_seq_len}"
    test_params = {
        "video_seq_len": video_seq_len,
        "pretrained_video_vqvae_model_key": vqvae_model_key,
    }
    kwargs = {**_model_params, **test_params}
    n_head = kwargs["n_head"]

    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[5, 6, 7]])
    attn_mask = torch.tril(torch.ones(7, 7)).unsqueeze(0)  # (b, seq_len, seq_len)
    head_mask = torch.ones(1, n_head, 7, 7)  # (b, h, seq_len, seq_len)

    model = model_fn(**kwargs)
    model.eval()
    num_tokens = model.num_in_tokens + model.num_out_tokens
    logits_mask = torch.ones(1, 7, num_tokens)  # (b, seq_len, num_tokens)

    out = model(x, y, attn_mask=attn_mask, head_mask=head_mask, logits_mask=logits_mask)
    actual = out.decoder_output.last_hidden_states
    assert_expected(actual.shape, (1, 7, 768))
    assert_expected(actual.sum().item(), expected, rtol=1, atol=1e-4)


@pytest.mark.parametrize(
    "video_seq_len, expected",
    [(8, 1520.8452), (16, -2085.2417), (32, -5190.5591)],
)
def test_forward_gpt_pretrained(model_fn, video_seq_len, expected):
    gpt_model_key = f"mugen_L{video_seq_len}"
    test_params = {
        "video_seq_len": video_seq_len,
        "pretrained_text_video_gpt_model_key": gpt_model_key,
    }
    kwargs = {**_model_params, **test_params}
    n_head = kwargs["n_head"]

    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[5, 6, 7]])
    attn_mask = torch.tril(torch.ones(7, 7)).unsqueeze(0)  # (b, seq_len, seq_len)
    head_mask = torch.ones(1, n_head, 7, 7)  # (b, h, seq_len, seq_len)

    model = model_fn(**kwargs)
    model.eval()
    num_tokens = model.num_in_tokens + model.num_out_tokens
    logits_mask = torch.ones(1, 7, num_tokens)  # (b, seq_len, num_tokens)

    out = model(x, y, attn_mask=attn_mask, head_mask=head_mask, logits_mask=logits_mask)
    actual = out.decoder_output.last_hidden_states
    assert_expected(actual.shape, (1, 7, 768))
    assert_expected(actual.sum().item(), expected, rtol=1, atol=1e-4)
