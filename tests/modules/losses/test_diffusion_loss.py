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
from torchmultimodal.modules.diffusion.schedules import (
    DiffusionSchedule,
    linear_beta_schedule,
)
from torchmultimodal.modules.losses.diffusion import DiffusionHybridLoss, VLBLoss
from torchmultimodal.utils.diffusion_utils import DiffusionOutput


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def schedule():
    return DiffusionSchedule(linear_beta_schedule(1000))


@pytest.fixture
def input():
    outs = [torch.randn(2, 3, 4, 4) for _ in range(4)]
    return DiffusionOutput(*outs)


@pytest.fixture
def target():
    target = {}
    target["x0"] = torch.randn(2, 3, 4, 4)
    target["xt"] = torch.randn(2, 3, 4, 4)
    target["noise"] = torch.randn(2, 3, 4, 4)
    target["t"] = torch.randint(0, 1000, (2,), dtype=torch.long)
    return target


# All expected values come after first testing the HybridLoss has the exact output
# as the corresponding p_losses in D2Go Guassian Diffusion
class TestDiffusionHybridLoss:
    @pytest.fixture
    def loss(self, schedule):
        simple = nn.MSELoss()
        return DiffusionHybridLoss(schedule, simple)

    def test_forward(self, loss, input, target):
        x0, xt, _, t = target.values()
        actual = loss(
            input.prediction, target["noise"], input.mean, input.log_variance, x0, xt, t
        )
        expected = torch.tensor(1.9727)
        assert_expected(actual, expected, rtol=0, atol=1e-4)


# All expected values come after first testing the HybridLoss has the exact output
# as the corresponding p_losses in D2Go Guassian Diffusion
class TestVLBLoss:
    @pytest.fixture
    def loss(self, schedule):
        return VLBLoss(schedule)

    def test_forward(self, loss, input, target):
        x0, xt, _, t = target.values()
        actual = loss(input.mean, input.log_variance, x0, xt, t)
        expected = torch.tensor(3.9603)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_approx_standard_normal_cdf(self, loss, input):
        x = input.prediction
        actual = loss.approx_standard_normal_cdf(x).mean()
        expected = torch.tensor(0.5051)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_discretized_gaussian_log_likelihood(self, loss, input, target):
        x = target["x0"]
        mean, log_var = input.mean, input.log_variance
        actual = loss.discretized_gaussian_log_likelihood(
            x, mean=mean, log_scale=log_var
        ).mean()
        expected = torch.tensor(-8.1964)
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_normal_kl(self, loss, input):
        mean, log_var = input.mean, input.log_variance
        actual = loss.normal_kl(mean, log_var, mean, log_var).mean()
        expected = torch.tensor(0.0)
        assert_expected(actual, expected, rtol=0, atol=1e-4)
