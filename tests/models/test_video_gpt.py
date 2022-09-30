# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected, set_rng_seed

from torchmultimodal.models.video_gpt import video_gpt, video_vqvae


@pytest.fixture(autouse=True)
def set_seed():
    return set_rng_seed(4)


_model_params = {
    "video_gpt": {
        "input_shape": (16, 64, 64),
        "latent_shape": (8, 32, 32),
        "d_model": 576,
        "n_head": 4,
        "dropout": 0.2,
        "attn_dropout": 0.3,
        "num_decoder_layers": 16,
        "use_gpt_init": True,
    },
    "video_vqvae": {
        "input_shape": (16, 64, 64),
        "conv_filter_sizes": ((4, 4, 4),),
        "conv_filter_strides": ((2, 2, 2),),
        "encoder_filter_size": (3, 3, 3),
        "encoder_filter_stride": (1, 1, 1),
        "in_channel_dim": 3,
        "encoder_hidden_dim": 240,
        "n_res_layers": 4,
        "attn_hidden_dim": 240,
        "num_embeddings": 1024,
        "embedding_dim": 256,
        "decoder_hidden_dim": 240,
    },
}


class TestVideoGPT:

    _model_name = "video_gpt"

    @pytest.fixture
    def model_params(self):
        return _model_params.get(self._model_name, {})

    @pytest.fixture
    def model_fn(self):
        return video_gpt

    def test_encode(self, model_fn, model_params):
        model = model_fn(**model_params)
        model.eval()

        x = torch.randn((1, 3, *model_params["input_shape"]))  # (b, c, *input_shape)

        actual = model.encode(x, "in")
        assert_expected(actual.shape, (1, 8192))
        assert_expected(actual.sum().item(), 6678187)

    def test_decode(self, model_fn, model_params):
        model = model_fn(**model_params)
        model.eval()

        latent_seq_len = torch.prod(torch.tensor(model_params["latent_shape"])).item()
        x = torch.randint(0, 10, (1, latent_seq_len))  # tokens

        actual = model.decode(x)
        assert_expected(actual.shape, (1, 3, 16, 64, 64))
        assert_expected(actual.sum().item(), 14629.2432, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize(
        "modality, expected_shape, expected_sum",
        [("in", (1, 2, 256), 38.8214), ("out", (1, 2, 256), -23.4659)],
    )
    def test_lookup(
        self, model_fn, model_params, modality, expected_shape, expected_sum
    ):
        model = model_fn(**model_params)
        model.eval()

        x = torch.tensor([[1, 2]])  # tokens

        actual = model.lookup(x, modality)
        assert_expected(actual.shape, expected_shape)
        assert_expected(actual.sum().item(), expected_sum, rtol=1e-5, atol=1e-4)

    def test_forward(self, model_fn, model_params):
        model = model_fn(**model_params)
        model.eval()

        n_head = model_params["n_head"]

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
        # Tolerance is fairly high but between Mac and Linux (AWS) it looks like the resuts
        # are slightly different when rtol=1e-5
        assert_expected(actual.sum().item(), 64.0230, rtol=1, atol=1e-4)


def test_video_vqvae():
    model_name = "video_vqvae"
    kwargs = _model_params.get(model_name, {})
    input_shape = kwargs.pop("input_shape")

    model = video_vqvae(**kwargs)
    model.eval()

    x = torch.randn((1, 3, *input_shape))
    out = model(x)
    actual = out.decoded
    assert_expected(actual.shape, (1, 3, 16, 64, 64))
    assert_expected(actual.sum().item(), -44372.4180, rtol=1, atol=1e-4)
