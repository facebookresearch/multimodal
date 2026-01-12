# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.models.adm_unet.attention_block import (
    ADMAttentionBlock,
)
from torchmultimodal.diffusion_labs.models.adm_unet.res_block import (
    adm_res_block,
    adm_res_downsample_block,
    adm_res_upsample_block,
    ADMResBlock,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def params():
    in_dim = 32  # in and out dim must be divisible by 32 because of GroupNorm
    out_dim = 32
    time_dim = 2
    kernel = 6
    return in_dim, out_dim, time_dim, kernel


@pytest.fixture
def x(params):
    in_dim, _, _, kernel = params
    out = torch.randn((1, in_dim, kernel, kernel), dtype=torch.float32)
    return out


@pytest.fixture
def t(params):
    time_dim = params[2]
    return torch.ones((1, time_dim), dtype=torch.float32)


# All expected values come after first testing the ADMResBlock has the exact output
# as the corresponding residual block class from ADM authors, then simply forward passing
# ADMResBlock with params, random seed, and initialization order in this file.
class TestADMResBlock:
    @pytest.fixture
    def block(self, params):
        in_dim, out_dim, time_dim, _ = params
        return partial(ADMResBlock, in_dim, out_dim, time_dim)

    def test_forward(self, block, x, t):
        res = block()
        actual = res(x, t)
        expected = torch.tensor(-193.9132)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_forward_no_scale_shift(self, block, x, t):
        res = block(scale_shift_conditional=False)
        actual = res(x, t)
        expected = torch.tensor(-96.7350)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_forward_rescale_skip(self, block, x, t):
        res = block(rescale_skip_connection=True)
        actual = res(x, t)
        expected = torch.tensor(-137.1381)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_channel_mismatch_error(self):
        with pytest.raises(ValueError):
            _ = ADMResBlock(1, 2, 3)

    def test_channel_indivisible_norm_group_error(self):
        with pytest.raises(ValueError):
            _ = ADMResBlock(2, 2, 3)

    def test_both_updownsample_error(self, block):
        with pytest.raises(ValueError):
            _ = block(use_upsample=True, use_downsample=True)


class TestADMAttentionBlock:
    @pytest.fixture
    def xattn_dim(self):
        return 48

    @pytest.fixture
    def cond(self, xattn_dim):
        return torch.ones((1, 4, xattn_dim), dtype=torch.float32)

    @pytest.fixture
    def block(self, params, xattn_dim):
        in_dim = params[0]
        return partial(ADMAttentionBlock, in_dim, xattn_dim, 1)

    def test_forward(self, block, x, cond):
        attn = block()
        actual = attn(x, cond)
        expected = torch.tensor(-53.3069)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_forward_rescale_skip(self, block, x, cond):
        attn = block(rescale_skip_connection=True)
        actual = attn(x, cond)
        expected = torch.tensor(-37.6994)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_channel_indivisible_norm_group_error(self):
        with pytest.raises(ValueError):
            _ = ADMAttentionBlock(2)


def test_adm_res_block(params, x, t):
    in_dim, out_dim, time_dim, _ = params
    res = adm_res_block(in_dim, out_dim, time_dim)
    actual = res(x, t)
    expected = torch.tensor(-193.9132)
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)


def test_adm_res_upsample_block(params, x, t):
    in_dim, _, time_dim, kernel = params
    res = adm_res_upsample_block(in_dim, time_dim)
    actual = res(x, t)
    expected = torch.tensor(-761.6157)
    expected_shape = torch.Size([1, in_dim, kernel * 2, kernel * 2])
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
    assert_expected(actual.shape, expected_shape)


def test_adm_res_downsample_block(params, x, t):
    in_dim, _, time_dim, kernel = params
    res = adm_res_downsample_block(in_dim, time_dim)
    actual = res(x, t)
    expected = torch.tensor(-36.9527)
    expected_shape = torch.Size([1, in_dim, kernel // 2, kernel // 2])
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
    assert_expected(actual.shape, expected_shape)


def test_adm_res_skipconv_block(params, x, t):
    in_dim, out_dim, time_dim, kernel = params
    res = adm_res_block(in_dim, out_dim * 2, time_dim)
    actual = res(x, t)
    expected = torch.tensor(-43.1194)
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
