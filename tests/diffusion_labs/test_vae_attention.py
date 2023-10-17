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
from torchmultimodal.diffusion_labs.models.vae.attention import (
    attention_res_block,
    AttentionResBlock,
    VanillaAttention,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(1234)


@pytest.fixture
def channels():
    return 64


@pytest.fixture
def norm_groups():
    return 16


@pytest.fixture
def norm_eps():
    return 1e-05


@pytest.fixture
def x(channels):
    bsize = 2
    height = 16
    width = 16
    return torch.randn(bsize, channels, height, width)


class TestVanillaAttention:
    @pytest.fixture
    def attn(self, channels):
        return VanillaAttention(channels)

    def test_forward(self, x, attn):
        actual = attn(x)
        expected = torch.tensor(32.0883)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        assert_expected(actual.shape, x.shape)


class TestAttentionResBlock:
    @pytest.fixture
    def attn(self, channels, norm_groups, norm_eps):
        return AttentionResBlock(
            channels,
            attn_module=nn.Identity(),
            norm_groups=norm_groups,
            norm_eps=norm_eps,
        )

    def test_forward(self, x, attn):
        actual = attn(x)
        expected = torch.tensor(295.1067)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
        assert_expected(actual.shape, x.shape)

    def test_channel_indivisible_norm_group_error(self):
        with pytest.raises(ValueError):
            _ = AttentionResBlock(64, nn.Identity(), norm_groups=30)


def test_attention_res_block(channels, x):
    attn = attention_res_block(channels)
    expected = torch.tensor(69.692)
    actual = attn(x)
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
    assert_expected(actual.shape, x.shape)
