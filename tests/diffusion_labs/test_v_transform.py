#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.diffusion_labs.transforms.diffusion_transform import (
    RandomDiffusionSteps,
)
from torchmultimodal.diffusion_labs.transforms.v_transform import ComputeV


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


class DummySchedule:
    def sample_steps(self, x):
        return x

    def sample_noise(self, x):
        return x

    def q_sample(self, x, noise, t):
        return x

    def __call__(self, var_name, t, shape):
        return torch.ones(shape)


def test_compute_v():
    schedule = DummySchedule()
    transform = nn.Sequential(RandomDiffusionSteps(schedule), ComputeV(schedule))
    actual = transform({"x": torch.ones(1)})["v"].mean()
    expected = torch.tensor(0.0)
    assert_expected(actual, expected, rtol=0, atol=1e-4)
