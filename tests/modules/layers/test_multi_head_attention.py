# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected
from torch import nn
from torchmultimodal.modules.layers.multi_head_attention import MultiHeadSelfAttention


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
