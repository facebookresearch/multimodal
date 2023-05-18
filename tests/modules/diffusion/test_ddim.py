#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.diffusion.ddim import DDIModule
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
        self.net = nn.Conv2d(3, 6, 3, 1, padding=1)

    def forward(self, x, t, c):
        x = self.net(x)
        return DiffusionOutput(prediction=x.chunk(2, dim=1)[0])


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


class TestDDIModule:
    @pytest.fixture
    def module(self):
        model = DummyUNet(True)
        schedule = DiffusionSchedule(linear_beta_schedule(1000))
        predictor = NoisePredictor(schedule)
        model = DDIModule(model, schedule, predictor)
        return model

    @pytest.fixture
    def input(self):
        data = {}
        data["prediction"] = torch.randn(2, 3, 4, 4)
        data["xt"] = torch.randn(2, 3, 4, 4)
        data["t"] = torch.randint(0, 1000, (2,), dtype=torch.long)
        data["cur_step"] = 9 * torch.ones(data["xt"].size(0), dtype=torch.long)
        data["next_step"] = 8 * torch.ones(data["xt"].size(0), dtype=torch.long)
        return data

    def test_remove_noise(self, module, input):
        set_rng_seed(4)
        xt = input["xt"]
        with torch.no_grad():
            actual = (
                module.remove_noise(xt, None, input["cur_step"], input["next_step"])
                .mean()
                .item()
            )

        expected = -0.1227
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_generator(self, module, input):
        set_rng_seed(4)
        xt = input["xt"]
        with torch.no_grad():
            gen = module.generator(xt)
            for i in range(10):
                actual = next(gen)
        expected = -0.1958
        assert_expected(actual.mean().item(), expected, rtol=0, atol=1e-4)
