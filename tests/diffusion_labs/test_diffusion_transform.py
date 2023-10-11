#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchmultimodal.diffusion_labs.transforms.diffusion_transform import (
    RandomDiffusionSteps,
)


class DummySchedule:
    def sample_steps(self, x):
        return x

    def sample_noise(self, x):
        return x

    def q_sample(self, x, noise, t):
        return x


def test_random_diffusion_steps():
    transform = RandomDiffusionSteps(DummySchedule())
    actual = len(transform(torch.ones(1)))
    expected = 4
    assert actual == expected, "Transform not returning correct keys"
