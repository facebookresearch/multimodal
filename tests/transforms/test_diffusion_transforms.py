#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from tests.test_utils import assert_expected
from torchmultimodal.transforms.diffusion_transforms import (
    Dalle2ImageTransform,
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


def test_dalle_image_transform():
    img_size = 5
    transform = Dalle2ImageTransform(image_size=img_size, image_min=-1, image_max=1)
    image = Image.new("RGB", size=(20, 20), color=(128, 0, 0))
    actual = transform(image).sum()
    normalized128 = 128 / 255 * 2 - 1
    normalized0 = -1
    expected = torch.tensor(
        normalized128 * img_size**2 + 2 * normalized0 * img_size**2
    )
    assert_expected(actual, expected, rtol=0, atol=1e-4)
