#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    cosine_beta_schedule,
    DiscreteGaussianSchedule,
    linear_beta_schedule,
    quadratic_beta_schedule,
    sigmoid_beta_schedule,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


# All expected values come after first testing the Schedule has the exact output
# as the corresponding q methods from GaussianDiffusion
class TestDiffusionSchedule:
    @pytest.fixture
    def module(self):
        schedule = DiscreteGaussianSchedule(linear_beta_schedule(1000))
        return schedule

    @pytest.fixture
    def input(self):
        input = {}
        input["x0"] = torch.randn(2, 3, 4, 4)
        input["xt"] = torch.randn(2, 3, 4, 4)
        input["noise"] = torch.randn(2, 3, 4, 4)
        input["t"] = torch.randint(0, 1000, (2,), dtype=torch.long)
        return input

    def test_sample_noise(self, module, input):
        x = input["x0"]
        actual = module.sample_noise(x).mean()
        expected = torch.tensor(-0.0291)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_sample_steps(self, module, input):
        x = input["x0"]
        actual = module.sample_steps(x).sum().item()
        expected = 1806
        assert actual == expected, "Wrong number of steps"

    def test_q_sample(self, module, input):
        actual = module.q_sample(input["x0"], input["noise"], input["t"]).mean()
        expected = torch.tensor(-0.0033)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_q_posterior(self, module, input):
        actual_mean, actual_log_var = module.q_posterior(
            input["x0"], input["xt"], input["t"]
        )
        expected_mean, expected_log_var = torch.tensor(-0.1944), torch.tensor(-4.6365)
        assert_expected(actual_mean.mean(), expected_mean, rtol=0, atol=1e-4)
        assert_expected(actual_log_var.mean(), expected_log_var, rtol=0, atol=1e-4)

    def test_steps(self, module):
        actual = module.steps
        expected = 1000
        assert actual == expected, "Wrong number of steps"

    def test_call(self, module, input):
        t, shape = input["t"], input["x0"].shape
        alphas_cumprod = module("alphas_cumprod", t, shape)
        expected_mean = torch.tensor(0.2016, dtype=torch.float64)
        expected_shape = (2, 1, 1, 1)
        assert_expected(alphas_cumprod.mean(), expected_mean, rtol=0, atol=1e-4)
        assert alphas_cumprod.shape == expected_shape, "Wrong shape"


def test_cosine_beta_schedule():
    actual = cosine_beta_schedule(1000).mean()
    expected = torch.tensor(0.0124415, dtype=torch.float64)
    assert_expected(actual, expected, rtol=0, atol=1e-6)


def test_linear_beta_schedule():
    actual = linear_beta_schedule(1000).mean()
    expected = torch.tensor(0.0100500, dtype=torch.float64)
    assert_expected(actual, expected, rtol=0, atol=1e-6)


def test_quadratic_beta_schedule():
    actual = quadratic_beta_schedule(1000).mean()
    expected = torch.tensor(0.0071743, dtype=torch.float64)
    assert_expected(actual, expected, rtol=0, atol=1e-6)


def test_sigmoid_beta_schedule():
    actual = sigmoid_beta_schedule(1000).mean()
    expected = torch.tensor(0.0100500, dtype=torch.float64)
    assert_expected(actual, expected, rtol=0, atol=1e-6)
