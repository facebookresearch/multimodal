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
    Generation,
    get_logits_mask,
    top_k_top_p_filtering,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def test_get_logits_mask():
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


class TestGeneration:
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

    def test_model_eval(self, model_fn):
        model = model_fn(**self._model_params)
        with pytest.warns(UserWarning):
            generator = Generation(max_seq_len=100, model=model)

    def test_sample(self, model_fn, mocker):
        # prevents the generation of tokens out side of codebook codes as the model
        # is not pre-trained so any token id is possible
        mocker.patch(
            "torchmultimodal.utils.generate.torch.multinomial",
            return_value=torch.tensor([[0]]),
        )
        input_shape = self._model_params["input_shape"]
        latent_shape = self._model_params["latent_shape"]
        latent_seq_len = torch.prod(torch.tensor(latent_shape)).item()
        model = model_fn(**self._model_params)
        model.eval()
        generator = Generation(max_seq_len=latent_seq_len, model=model)
        x = torch.randn(1, 3, *input_shape)  # (b, c, *input_shape)
        actual = generator.sample(x, use_cache=True, causal=True)
        assert_expected(
            actual.shape, torch.Size([1, 3, 4, 8, 8])
        )  # (b, c, *input_shape)
        assert_expected(actual.sum().item(), -41.6888, rtol=1e-4, atol=1e-5)


class TestTopKTopPFiltering:
    _func_params = {
        "top_k": 0,
        "top_p": 1.0,
        "filter_value": 0.0,
        "min_tokens_to_keep": 1,
    }

    def test_min_tokens_to_keep(self):
        kwargs = {**self._func_params, **{"top_k": 1, "min_tokens_to_keep": 2}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = top_k_top_p_filtering(logits, **kwargs)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 9.0]])
        assert_expected(actual, expected)

    def test_top_k_invalid(self):
        kwargs = {**self._func_params, **{"top_k": -1}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        with pytest.raises(ValueError):
            top_k_top_p_filtering(logits, **kwargs)

    def test_top_p_invalid(self):
        kwargs = {**self._func_params, **{"top_p": 2.0}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        with pytest.raises(ValueError):
            top_k_top_p_filtering(logits, **kwargs)

        kwargs = {**self._func_params, **{"top_p": -1.0}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        with pytest.raises(ValueError):
            top_k_top_p_filtering(logits, **kwargs)

    def test_top_k_default(self):
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = top_k_top_p_filtering(logits)
        expected = logits
        assert_expected(actual, expected)

    def test_top_p_default(self):
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = top_k_top_p_filtering(logits)
        expected = logits
        assert_expected(actual, expected)

    def test_top_k(self):
        kwargs = {**self._func_params, **{"top_k": 5}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = top_k_top_p_filtering(logits, **kwargs)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        assert_expected(actual, expected)

    def test_top_p(self):
        kwargs = {**self._func_params, **{"top_p": 0.9}}
        logits = torch.arange(10, dtype=torch.float).unsqueeze(0)
        actual = top_k_top_p_filtering(logits, **kwargs)
        expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0]])
        assert_expected(actual, expected)
