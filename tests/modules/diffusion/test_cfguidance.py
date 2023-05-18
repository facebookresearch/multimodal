# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.diffusion.cfguidance import CFGuidance
from torchmultimodal.utils.diffusion_utils import DiffusionOutput


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def params():
    in_channels = 3
    s = 4
    embed_dim = 6
    return in_channels, s, embed_dim


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
    def cond(self, params):
        embed_dim = params[-1]
        c = torch.ones(1, embed_dim)
        return {"test": c}

    @pytest.fixture
    def dim_cond(self, cond):
        return {k: v.shape[-1] for k, v in cond.items()}

    @pytest.fixture
    def models(self):
        return {"mean": DummyADM(False), "mean_variance": DummyADM(True)}

    @pytest.fixture
    def cfguidance_models(self, models, dim_cond):
        return {
            k: CFGuidance(model, dim_cond, p=0, guidance=2, learn_null_emb=False)
            for k, model in models.items()
        }

    def test_training_forward(self, cfguidance_models, params, cond):
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim, params[1], params[1])
        t = torch.ones(1, 1)
        for k, cfguidance_model in cfguidance_models.items():
            actual = cfguidance_model(x, t, cond)
            if k == "mean":  # if adm model returns only prediction
                expected = 4 * torch.ones(1, embed_dim, params[1], params[1])
                assert_expected(actual.prediction, expected)
                assert_expected(actual.variance_value, None)
            elif (
                k == "mean_variance"
            ):  # if adm model returns both prediction and variance
                expected_prediction = 4 * torch.ones(1, embed_dim, params[1], params[1])
                expected_variance = torch.ones(1, embed_dim, params[1], params[1])
                assert_expected(actual.prediction, expected_prediction)
                assert_expected(actual.variance_value, expected_variance)

    def test_inference_forward(self, cfguidance_models, params, cond):
        embed_dim = params[-1]
        x = torch.ones(1, embed_dim, params[1], params[1])
        t = torch.ones(1, 1)

        for k, cfguidance_model in cfguidance_models.items():
            cfguidance_model.eval()
            actual = cfguidance_model(x, t, cond)
            if k == "mean":  # if adm model returns only prediction
                expected_prediction = 6 * torch.ones(1, embed_dim, params[1], params[1])
                assert_expected(actual.prediction, expected_prediction)
                assert_expected(actual.variance_value, None)
            elif (
                k == "mean_variance"
            ):  # if adm model returns both prediction and variance
                expected_prediction = 6 * torch.ones(1, embed_dim, params[1], params[1])
                expected_variance = torch.ones(1, embed_dim, params[1], params[1])
                assert_expected(actual.prediction, expected_prediction)
                assert_expected(actual.variance_value, expected_variance)

    def test_get_prob_dict(self, cfguidance_models):
        cfguidance_model = cfguidance_models["mean"]
        actual = cfguidance_model._get_prob_dict(0.1)
        expected = {"test": 0.1}
        assert_expected(actual, expected)

        actual = cfguidance_model._get_prob_dict({"test": 0.1})
        expected = {"test": 0.1}
        assert_expected(actual, expected)

        with pytest.raises(ValueError):
            actual = cfguidance_model._get_prob_dict({"test_2": 0.1, "test": 0.1})

        with pytest.raises(TypeError):
            actual = cfguidance_model._get_prob_dict("test")
