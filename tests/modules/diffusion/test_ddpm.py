#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.diffusion.ddpm import DDPModule
from torchmultimodal.modules.diffusion.predictors import NoisePredictor
from torchmultimodal.modules.diffusion.schedules import (
    DiffusionSchedule,
    linear_beta_schedule,
)
from torchmultimodal.utils.diffusion_utils import DiffusionOutput


class DummyUNet(nn.Module):
    def __init__(self, learn_variance=True):
        super().__init__()
        self.learn_variance = learn_variance
        channels = 6 if learn_variance else 3
        self.net = nn.Conv2d(3, channels, 3, 1, padding=1)

    def forward(self, x, t, c):
        x = F.tanh(self.net(x))
        var_value = None
        if self.learn_variance:
            x, var_value = x.chunk(2, dim=1)
            var_value = (var_value + 1) / 2
        return DiffusionOutput(prediction=x, variance_value=var_value)


# All expected values come after first testing the Schedule has the exact output
# as the corresponding p methods from GaussianDiffusion in D2Go
class TestDDPModule:
    @pytest.fixture
    def module(self):
        set_rng_seed(4)
        model = DummyUNet(True)
        schedule = DiffusionSchedule(linear_beta_schedule(1000))
        predictor = NoisePredictor(schedule)
        eval_steps = torch.arange(0, 1000, 50)
        model = DDPModule(model, schedule, predictor, eval_steps)
        return model

    @pytest.fixture
    def input(self):
        set_rng_seed(4)
        sample = {}
        sample["prediction"] = torch.randn(2, 3, 4, 4)
        sample["xt"] = torch.randn(2, 3, 4, 4)
        sample["t"] = torch.randint(0, 1000, (2,), dtype=torch.long)
        return sample

    def test_predict_parameters(self, module, input):
        inp = DiffusionOutput(prediction=input["prediction"])
        xt, t = input["xt"], input["t"]
        actual_mean, actual_log_var = module.predict_parameters(inp, xt, t)
        expected_mean, expected_log_var = torch.tensor(-0.1970), torch.tensor(-5.3342)
        assert_expected(actual_mean.mean(), expected_mean, rtol=0, atol=1e-4)
        assert_expected(actual_log_var.mean(), expected_log_var, rtol=0, atol=1e-4)

    def test_remove_noise(self, module, input):
        set_rng_seed(4)
        xt, t = input["xt"], input["t"]
        with torch.no_grad():
            actual = module.remove_noise(xt, t, None).mean()
        expected = torch.tensor(-0.1876)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_generator(self, module, input):
        set_rng_seed(4)
        xt = input["xt"]
        module.eval()
        with torch.no_grad():
            gen = module.generator(xt)
            for actual in gen:
                pass
        expected = torch.tensor(-20.1526)
        assert_expected(actual.mean(), expected, rtol=0, atol=1e-4)

    def test_forward(self, module, input):
        xt, t = input["xt"], input["t"]
        module.train()
        with torch.no_grad():
            actual = module(xt, t)
        expected_pred, expected_var_value = torch.tensor(-0.0852), torch.tensor(0.4979)
        expected_mean, expected_log_var = torch.tensor(-0.1955), torch.tensor(-5.3285)
        assert_expected(actual.prediction.mean(), expected_pred, rtol=0, atol=1e-4)
        assert_expected(
            actual.variance_value.mean(), expected_var_value, rtol=0, atol=1e-4
        )
        assert_expected(actual.mean.mean(), expected_mean, rtol=0, atol=1e-4)
        assert_expected(actual.log_variance.mean(), expected_log_var, rtol=0, atol=1e-4)
