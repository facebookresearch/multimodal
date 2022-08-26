# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected, set_rng_seed

from torchmultimodal.models.video_gpt import video_gpt
from torchmultimodal.utils.generate import (
    GenerationUtil,
    get_logits_mask,
    LogitsFilter,
    SampleOutput,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


class TestLogitsMask:
    def test_normal(self):
        actual = get_logits_mask(
            in_seq_len=3, out_seq_len=4, num_in_tokens=4, num_out_tokens=6
        )
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        assert_expected(actual, expected)

    def test_zero_dims(self):
        actual = get_logits_mask(
            in_seq_len=0, out_seq_len=0, num_in_tokens=0, num_out_tokens=0
        )
        assert actual.nelement() == 0

    def test_in_seq_only(self):
        actual = get_logits_mask(
            in_seq_len=1, out_seq_len=0, num_in_tokens=4, num_out_tokens=6
        )
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        assert_expected(actual, expected)

    def test_out_seq_only(self):
        actual = get_logits_mask(
            in_seq_len=0, out_seq_len=1, num_in_tokens=4, num_out_tokens=6
        )
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        assert_expected(actual, expected)


class TestGenerationUtil:
    _model_params = {
        "input_shape": (4, 8, 8),
        "latent_shape": (2, 4, 4),
        "d_model": 576,
        "n_head": 4,
        "num_decoder_layers": 16,
        "dropout": 0.2,
        "attn_dropout": 0.3,
    }

    @pytest.fixture
    def model_fn(self):
        return video_gpt

    @pytest.fixture
    def generation_model(self, model_fn):
        model = model_fn(**self._model_params)
        model.eval()
        return GenerationUtil(model=model)

    def test_model_eval_warning(self, model_fn):
        model = model_fn(**self._model_params)
        with pytest.warns(UserWarning):
            generator = GenerationUtil(model=model)

    def test_sample(self, generation_model, mocker):
        # prevents the generation of tokens out side of codebook codes as the model
        # is not pre-trained so any token id is possible
        mocker.patch(
            "torchmultimodal.utils.generate.torch.multinomial",
            return_value=torch.tensor([[0]]),
        )
        input_shape = self._model_params["input_shape"]
        latent_shape = self._model_params["latent_shape"]
        latent_seq_len = torch.prod(torch.tensor(latent_shape)).item()
        x = torch.randn(1, 3, *input_shape)  # (b, c, *input_shape)
        out = generation_model.sample(
            x, max_seq_len=latent_seq_len, use_cache=True, causal=True
        )
        assert isinstance(out, SampleOutput)
        actual = out.samples
        assert_expected(
            actual.shape, torch.Size([1, 3, 4, 8, 8])
        )  # (b, c, *input_shape)
        assert_expected(actual.sum().item(), -41.6888, rtol=1e-4, atol=1e-5)


class TestLogitsFilter:
    _func_params = {
        "top_k": 0,
        "top_p": 1.0,
        "filter_value": 0.0,
        "min_tokens_to_keep": 1,
    }

    @pytest.fixture
    def filter_fn(self):
        return LogitsFilter

    def test_min_tokens_to_keep(self, filter_fn):
        kwargs = {**self._func_params, **{"top_k": 1, "min_tokens_to_keep": 2}}
        logits_filter = filter_fn(**kwargs)
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = logits_filter(logits)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0]])
        assert_expected(actual, expected)

    def test_top_k_invalid(self, filter_fn):
        kwargs = {**self._func_params, **{"top_k": -1}}
        with pytest.raises(ValueError):
            logits_filter = filter_fn(**kwargs)

    def test_top_p_invalid(self, filter_fn):
        kwargs = {**self._func_params, **{"top_p": 2.0}}
        with pytest.raises(ValueError):
            logits_filter = filter_fn(**kwargs)

        kwargs = {**self._func_params, **{"top_p": -1.0}}
        with pytest.raises(ValueError):
            logits_filter = filter_fn(**kwargs)

    def test_default(self, filter_fn):
        kwargs = self._func_params
        logits_filter = filter_fn(**kwargs)
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = logits_filter(logits)
        expected = logits
        assert_expected(actual, expected)

    def test_top_k(self, filter_fn):
        kwargs = {**self._func_params, **{"top_k": 5}}
        logits_filter = filter_fn(**kwargs)
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = logits_filter(logits)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        assert_expected(actual, expected)

    def test_top_p(self, filter_fn):
        kwargs = {**self._func_params, **{"top_p": 0.9}}
        logits_filter = filter_fn(**kwargs)
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = logits_filter(logits)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0]])
        assert_expected(actual, expected)
