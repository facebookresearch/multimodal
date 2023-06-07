# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, init_weights_with_constant
from torch import nn
from torchmultimodal.modules.layers.multi_head_attention import (
    MultiHeadAttentionWithCache,
    MultiHeadSelfAttention,
)


class TestMultiHeadSelfAttention:
    @pytest.fixture
    def embed_dim(self):
        return 4

    @pytest.fixture
    def multi_head_self_attn(self, embed_dim):
        mhsa = MultiHeadSelfAttention(embed_dim, num_heads=2)
        mhsa.input_proj.weight = nn.Parameter(torch.ones(3 * embed_dim, embed_dim))
        mhsa.input_proj.bias = nn.Parameter(torch.ones(3 * embed_dim))
        mhsa.output_proj.weight = nn.Parameter(torch.ones(embed_dim, embed_dim))
        mhsa.output_proj.bias = nn.Parameter(torch.ones(embed_dim))
        mhsa.eval()
        return mhsa

    def test_multi_head_self_attention(
        self,
        embed_dim,
        multi_head_self_attn,
    ):
        q = torch.Tensor([[[1, 2, 3, 1], [4, 3, 2, 1], [1, 1, 1, 1]]])
        actual = multi_head_self_attn(q)
        expected = torch.tensor(
            [
                [
                    [45.0, 45.0, 45.0, 45.0],
                    [45.0, 45.0, 45.0, 45.0],
                    [45.0, 45.0, 45.0, 45.0],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scripting(
        self,
        embed_dim,
        multi_head_self_attn,
    ):
        q = torch.Tensor([[[1, 2, 3, 1], [4, 3, 2, 1], [1, 1, 1, 1]]])
        scripted_model = torch.jit.script(multi_head_self_attn)
        assert_expected(scripted_model(q), multi_head_self_attn(q), rtol=0, atol=1e-4)


class TestMultiHeadAttentionWithCache:
    @pytest.fixture
    def dim_q(self):
        return 4

    @pytest.fixture
    def dim_kv(self):
        return 2

    @pytest.fixture
    def q(self):
        return torch.Tensor([[[1, 2, 3, 1], [4, 3, 2, 1], [1, 1, 1, 1]]])

    @pytest.fixture
    def current_key_value(self):
        return torch.Tensor(
            [
                [
                    [[8.0, 8.0], [11.0, 11.0], [5.0, 5.0]],
                    [[8.0, 8.0], [11.0, 11.0], [5.0, 5.0]],
                ]
            ]
        )

    @pytest.fixture
    def past_key_value(self):
        return torch.Tensor(
            [
                [
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                    [[7.0, 7.0], [9.0, 9.0], [4.0, 4.0]],
                ]
            ]
        )

    @pytest.fixture
    def multi_head_self_attn_use_cache(self, dim_q):
        mha = MultiHeadAttentionWithCache(dim_q, dim_q, num_heads=2, use_cache=True)
        init_weights_with_constant(mha)
        mha.eval()
        return mha

    @pytest.fixture
    def multi_head_cross_attn(self, dim_q, dim_kv):
        mha = MultiHeadAttentionWithCache(dim_q, dim_kv, num_heads=2)
        init_weights_with_constant(mha)
        mha.eval()
        return mha

    def test_multi_head_self_attention_use_cache(
        self,
        multi_head_self_attn_use_cache,
        current_key_value,
        past_key_value,
        q,
    ):
        actual = multi_head_self_attn_use_cache(
            q, q, q, past_key_value=(past_key_value, past_key_value)
        )
        expected = torch.tensor(
            [
                [
                    [45.0, 45.0, 45.0, 45.0],
                    [45.0, 45.0, 45.0, 45.0],
                    [45.0, 45.0, 45.0, 45.0],
                ]
            ]
        )
        assert_expected(actual.attn_output, expected, rtol=0, atol=1e-4)
        # Check that the cache is properly updated
        assert_expected(
            actual.past_key_value[0],
            torch.cat([past_key_value, current_key_value], dim=2),
        )
        assert_expected(
            actual.past_key_value[1],
            torch.cat([past_key_value, current_key_value], dim=2),
        )

    def test_multi_head_cross_attention(self, multi_head_cross_attn, q):
        kv = torch.Tensor(torch.Tensor([[[3, 2], [1, 1]]]))
        actual = multi_head_cross_attn(q, kv, kv)
        expected = torch.tensor(
            [
                [
                    [25.0, 25.0, 25.0, 25.0],
                    [25.0, 25.0, 25.0, 25.0],
                    [25.0, 25.0, 25.0, 25.0],
                ],
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scripting(
        self,
        multi_head_self_attn_use_cache,
        q,
    ):
        scripted_model = torch.jit.script(multi_head_self_attn_use_cache)
        assert_expected(
            scripted_model(q, q, q).attn_output,
            multi_head_self_attn_use_cache(q, q, q).attn_output,
            rtol=0,
            atol=1e-4,
        )

    def test_multi_head_cross_attention_invalid_input(self, multi_head_cross_attn, q):
        kv = torch.Tensor(torch.Tensor([[[3, 2]], [[1, 1]]]))
        with pytest.raises(ValueError):
            actual = multi_head_cross_attn(q, kv, kv)
