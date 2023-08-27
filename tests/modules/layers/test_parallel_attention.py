# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

# import torch
from tests.test_utils import assert_expected  # , init_weights_with_constant

# from torch import nn
from torchmultimodal.modules.layers.parallel_attention import ParallelAttentionBlock


class TestParallelAttentionBlocks:
    @pytest.fixture
    def embedding_dim(self):
        return 64

    @pytest.fixture
    def total_layers(self):
        return 1

    @pytest.fixture
    def mqa_num_heads(self):
        return 1

    @pytest.fixture
    def gqa_num_heads(self):
        return 2

    @pytest.fixture
    def num_heads(self):
        return 16

    @pytest.fixture
    def mha_parallel_attention(self, embedding_dim, num_heads, total_layers):
        print(f"{embedding_dim=}, {num_heads=}, {total_layers=}")
        pab_mha = ParallelAttentionBlock(
            emb_dimension=embedding_dim,
            num_heads=num_heads,
            use_group_query_attention=False,
            num_layers=total_layers,
            use_weight_init=True,
        )
        pab_mha.eval()
        return pab_mha

    @pytest.fixture
    def gqa_parallel_attention(
        self, embedding_dim, num_heads, total_layers, gqa_num_heads
    ):
        print(f"{embedding_dim=}, {num_heads=}, {total_layers=}")
        pab_gqa = ParallelAttentionBlock(
            emb_dimension=embedding_dim,
            num_heads=num_heads,
            use_group_query_attention=True,
            num_heads_group_query_attention=gqa_num_heads,
            num_layers=total_layers,
            use_weight_init=True,
        )
        pab_gqa.eval()
        return pab_gqa

    @pytest.fixture
    def mqa_parallel_attention(
        self, embedding_dim, num_heads, total_layers, mqa_num_heads
    ):
        print(f"{embedding_dim=}, {num_heads=}, {total_layers=}")
        pab_mqa = ParallelAttentionBlock(
            emb_dimension=embedding_dim,
            num_heads=num_heads,
            use_group_query_attention=True,
            num_heads_group_query_attention=mqa_num_heads,
            num_layers=total_layers,
            use_weight_init=True,
        )
        pab_mqa.eval()
        return pab_mqa

    def test_mha_parallel_attention(self, mha_parallel_attention, num_heads):
        # confirm all K and V keys match Q (i.e. MHA)
        assert_expected(num_heads, mha_parallel_attention.num_kv)

    def test_mqa_parallel_attention(
        self, mqa_parallel_attention, num_heads, mqa_num_heads
    ):
        print("in test")

        # confirm all K and V keys match MQA num heads (i.e. MQA == 1)
        assert_expected(mqa_num_heads, mqa_parallel_attention.num_kv)

    def test_gqa_parallel_attention(
        self, gqa_parallel_attention, num_heads, gqa_num_heads
    ):
        print("in test")

        # confirm all K and V keys match GQA num heads (i.e. GQA >= 2)
        assert_expected(gqa_num_heads, gqa_parallel_attention.num_kv)
