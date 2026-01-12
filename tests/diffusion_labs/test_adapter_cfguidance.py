# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Sequence, Tuple, Union

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.modules.adapters.cfguidance import CFGuidance
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def params() -> Tuple[int, int, int]:
    in_channels = 3
    s = 4
    embed_dim = 6
    return in_channels, s, embed_dim


@pytest.fixture
def x(params) -> Tensor:
    embed_dim = params[-1]
    return torch.ones(1, embed_dim, params[1], params[1])


@pytest.fixture
def t() -> Tensor:
    return torch.ones(1, 1)


class DummyADM(nn.Module):
    def __init__(self, variance_flag=False):
        super().__init__()
        self.variance_flag = variance_flag

    def forward(self, x, t, c):
        c_sum = 0
        count = 0
        for i, (k, v) in enumerate(c.items()):
            c_sum += v
            count += 1

        prediction = x + t[..., None, None] + c_sum[..., None, None] + count
        if self.variance_flag:
            variance_value = x
        else:
            variance_value = None
        return DiffusionOutput(prediction=prediction, variance_value=variance_value)


class TestCFGuidance:
    @pytest.fixture
    def dim_cond(self, params) -> Dict[str, Union[int, Sequence[int]]]:
        embed_dim = params[-1]
        return {"test": embed_dim, "test_list": [embed_dim]}

    @pytest.fixture
    def cond(self, dim_cond) -> Dict[str, Tensor]:
        val = {}
        for k, v in dim_cond.items():
            if isinstance(v, int):
                val[k] = torch.ones(1, v)
            else:
                val[k] = torch.ones(1, *v)
        return val

    @pytest.fixture
    def models(self):
        return {"mean": DummyADM(False), "mean_variance": DummyADM(True)}

    @pytest.fixture
    def cfguidance_models(self, models, dim_cond):
        return {
            k: CFGuidance(model, dim_cond, p=0, guidance=2, learn_null_emb=False)
            for k, model in models.items()
        }

    @pytest.fixture
    def cfguidance_models_p_1(self, models, dim_cond):
        return {
            k: CFGuidance(model, dim_cond, p=1, guidance=2, learn_null_emb=False)
            for k, model in models.items()
        }

    @pytest.fixture
    def cfguidance_models_train_embeddings(self, models, dim_cond, cond):
        train_embeddings = {k: v * 5 for k, v in cond.items()}
        return {
            k: CFGuidance(
                model,
                dim_cond,
                p=1,
                guidance=2,
                learn_null_emb=False,
                train_unconditional_embeddings=train_embeddings,
            )
            for k, model in models.items()
        }

    @pytest.fixture
    def cfguidance_models_eval_embeddings(self, models, dim_cond, cond):
        eval_embeddings = {k: v * 3 for k, v in cond.items()}
        return {
            k: CFGuidance(
                model,
                dim_cond,
                p=1,
                guidance=2,
                learn_null_emb=False,
                eval_unconditional_embeddings=eval_embeddings,
            )
            for k, model in models.items()
        }

    @pytest.mark.parametrize(
        "cfg_models,expected_multiplier",
        [
            ("cfguidance_models", 6),
            ("cfguidance_models_p_1", 4),
            ("cfguidance_models_train_embeddings", 14),
            ("cfguidance_models_eval_embeddings", 4),
        ],
    )
    def test_training_forward(
        self, cfg_models, x, t, cond, expected_multiplier, request
    ):
        cfguidance_models = request.getfixturevalue(cfg_models)
        for k, cfguidance_model in cfguidance_models.items():
            actual = cfguidance_model(x, t, cond)
            expected_mean = expected_multiplier * x
            expected_variance = x if k == "mean_variance" else None
            assert_expected(actual.prediction, expected_mean)
            assert_expected(actual.variance_value, expected_variance)

    @pytest.mark.parametrize(
        "cfg_models,expected_multiplier",
        [
            ("cfguidance_models", 10),
            ("cfguidance_models_p_1", 10),
            ("cfguidance_models_train_embeddings", -10),
            ("cfguidance_models_eval_embeddings", -2),
        ],
    )
    def test_inference_forward(
        self, cfg_models, x, t, cond, expected_multiplier, request
    ):
        cfguidance_models = request.getfixturevalue(cfg_models)
        for k, cfguidance_model in cfguidance_models.items():
            cfguidance_model.eval()
            actual = cfguidance_model(x, t, cond)
            expected_mean = expected_multiplier * x
            expected_variance = x if k == "mean_variance" else None
            assert_expected(actual.prediction, expected_mean)
            assert_expected(actual.variance_value, expected_variance)

    @pytest.mark.parametrize(
        "cfg_models,expected_multiplier",
        [
            ("cfguidance_models", 6),
            ("cfguidance_models_p_1", 6),
            ("cfguidance_models_train_embeddings", 6),
            ("cfguidance_models_eval_embeddings", 6),
        ],
    )
    def test_inference_0_guidance_forward(
        self, cfg_models, x, t, cond, expected_multiplier, request
    ):
        cfguidance_models = request.getfixturevalue(cfg_models)
        for k, cfguidance_model in cfguidance_models.items():
            cfguidance_model.guidance = 0
            cfguidance_model.eval()
            actual = cfguidance_model(x, t, cond)
            expected_mean = expected_multiplier * x
            expected_variance = x if k == "mean_variance" else None
            assert_expected(actual.prediction, expected_mean)
            assert_expected(actual.variance_value, expected_variance)

    @pytest.mark.parametrize(
        "cfg_models,expected_multiplier",
        [
            ("cfguidance_models", 4),
            ("cfguidance_models_p_1", 4),
            ("cfguidance_models_train_embeddings", 14),
            ("cfguidance_models_eval_embeddings", 10),
        ],
    )
    def test_inference_no_cond_forward(
        self, cfg_models, x, t, expected_multiplier, request
    ):
        cfguidance_models = request.getfixturevalue(cfg_models)
        for k, cfguidance_model in cfguidance_models.items():
            cfguidance_model.eval()
            actual = cfguidance_model(x, t, None)
            expected_mean = expected_multiplier * x
            expected_variance = x if k == "mean_variance" else None
            assert_expected(actual.prediction, expected_mean)
            assert_expected(actual.variance_value, expected_variance)

    def test_get_prob_dict(self, cfguidance_models):
        cfguidance_model = cfguidance_models["mean"]
        actual = cfguidance_model._get_prob_dict(0.1)
        expected = {"test": 0.1, "test_list": 0.1}
        assert_expected(actual, expected)

        actual = cfguidance_model._get_prob_dict({"test": 0.1, "test_list": 0.2})
        expected = {"test": 0.1, "test_list": 0.2}
        assert_expected(actual, expected)

        with pytest.raises(ValueError):
            actual = cfguidance_model._get_prob_dict({"test_2": 0.1, "test": 0.1})

        with pytest.raises(TypeError):
            actual = cfguidance_model._get_prob_dict("test")

    def test_gen_unconditional_embeddings(self, cfguidance_models, params, cond):
        cfguidance_model = cfguidance_models["mean"]
        actual = cfguidance_model._gen_unconditional_embeddings(
            None, torch.zeros, False
        )
        assert set(actual.keys()) == set(cond.keys())
        for p in actual.values():
            assert_expected(p.mean().item(), 0.0)

        actual = cfguidance_model._gen_unconditional_embeddings(
            cond, torch.zeros, False
        )
        assert set(actual.keys()) == set(cond.keys())
        for p in actual.values():
            assert_expected(p.mean().item(), 1.0)
