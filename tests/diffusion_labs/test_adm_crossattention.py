# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.diffusion_labs.modules.layers.attention_block import (
    adm_attention,
    ADMCrossAttention,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def params():
    embed_dim = 32  # in and out dim must be divisible by 32 because of GroupNorm
    cond_dim = 48
    n_head = 2
    kernel = 2
    return embed_dim, cond_dim, n_head, kernel


@pytest.fixture
def x(params):
    embed_dim, _, n_head, kernel = params
    out = torch.randn(
        (1, n_head, kernel, kernel, embed_dim // n_head), dtype=torch.float32
    )
    return out


@pytest.fixture
def c(params):
    cond_dim = params[1]
    return torch.ones((1, 4, cond_dim), dtype=torch.float32)


# All expected values come after first testing that ADMCrossAttention has
# the exact output as the corresponding QKVAttention class in ADM, then simply forward passing
# ADMCrossAttention with params, random seed, and initialization order in this file.
class TestADMCrossAttention:
    @pytest.fixture
    def attn(self, params):
        embed_dim, cond_dim, *_ = params
        return ADMCrossAttention(dim_qkv=embed_dim, dim_cond=cond_dim)

    def test_forward(self, attn, x, c):
        actual = attn(x, x, x, c)
        expected = torch.tensor(-18.1184)
        assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)

    def test_head_indivisible_error(self, attn, x):
        x_ = torch.cat([x, x[:, :1, :, :, :]], dim=1)
        with pytest.raises(ValueError):
            _ = attn(x_, x_, x_)

    def test_embedding_dim_error(self, attn, x):
        x_ = x[:, :, :, :, :-1]
        with pytest.raises(ValueError):
            _ = attn(x_, x_, x_)


def test_adm_attention(params, x, c):
    embed_dim, cond_dim, n_head, _ = params
    attn = adm_attention(
        num_channels=embed_dim,
        dim_cond=cond_dim,
        num_heads=n_head,
    )
    x = x.permute(0, 2, 3, 4, 1).flatten(start_dim=-2, end_dim=-1)
    actual = attn(x, conditional_embedding=c)
    expected = torch.tensor(3.6252)
    assert_expected(actual.sum(), expected, rtol=0, atol=1e-4)
