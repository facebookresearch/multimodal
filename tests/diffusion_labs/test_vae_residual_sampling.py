#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.models.vae.residual_sampling import (
    Downsample2D,
    Upsample2D,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(54321)


@pytest.fixture
def in_channels():
    return 2


@pytest.fixture
def x(in_channels):
    bsize = 2
    height = 16
    width = 16
    return torch.randn(bsize, in_channels, height, width)


def test_upsample(in_channels, x):
    upsampler = Upsample2D(channels=in_channels)
    actual = upsampler(x)
    expected = torch.tensor(-350.5232)
    assert_expected(
        actual.size(),
        torch.Size(
            [
                x.size(0),
                x.size(1),
                x.size(2) * 2,
                x.size(3) * 2,
            ]
        ),
    )
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)


class TestDownsample:
    @pytest.fixture
    def downsampler_fn(self, in_channels):
        return partial(Downsample2D, channels=in_channels)

    @pytest.mark.parametrize(
        "asymmetric_padding,expected_value", [(True, -18.3393), (False, -28.8385)]
    )
    def test_downsample(self, downsampler_fn, x, asymmetric_padding, expected_value):
        downsampler = downsampler_fn(asymmetric_padding=asymmetric_padding)
        actual = downsampler(x)
        expected = torch.tensor(expected_value)
        assert_expected(
            actual.size(),
            torch.Size(
                [
                    x.size(0),
                    x.size(1),
                    x.size(2) // 2,
                    x.size(3) // 2,
                ]
            ),
        )
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
