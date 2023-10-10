#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.predictors.noise_predictor import NoisePredictor
from torchmultimodal.diffusion_labs.predictors.target_predictor import TargetPredictor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
    linear_beta_schedule,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def input():
    data = {}
    data["prediction"] = torch.randn(2, 3, 4, 4)
    data["xt"] = torch.randn(2, 3, 4, 4)
    data["t"] = torch.randint(0, 1000, (2,), dtype=torch.long)
    return data


# All expected values come after first testing the Schedule has the exact output
# as the corresponding q methods from GaussianDiffusion
class TestNoisePredictor:
    @pytest.fixture
    def module(self):
        schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))
        predictor = NoisePredictor(schedule, None)
        return predictor

    def test_predict_x0(self, module, input):
        actual = module.predict_x0(**input).mean()
        expected = torch.tensor(-1.9352)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_predict_noise(self, module, input):
        actual = module.predict_noise(**input).mean()
        expected = torch.tensor(0.0411)
        assert_expected(actual, expected, rtol=0, atol=1e-4)


class TestTargetPredictor:
    @pytest.fixture
    def module(self):
        schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))
        predictor = TargetPredictor(schedule, None)
        return predictor

    def test_predict_x0(self, module, input):
        actual = module.predict_x0(**input).mean()
        expected = torch.tensor(0.0411)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_predict_noise(self, module, input):
        # TODO: add with DDIM
        pass
