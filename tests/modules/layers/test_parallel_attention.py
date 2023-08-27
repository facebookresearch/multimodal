# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.test_utils import assert_expected  # , init_weights_with_constant
from torchmultimodal.modules.layers.parallel_attention import ParallelAttentionBlock


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(2023)


class TestParallelAttentionBlocks:
    @pytest.fixture
    def embedding_dim(self):
        return 16

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
    def max_seq_len(self):
        return 32

    @pytest.fixture
    def mha_parallel_attention(self, embedding_dim, num_heads, total_layers):
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

    def test_mha_parallel_attention(
        self,
        mha_parallel_attention,
        num_heads,
        embedding_dim,
        max_seq_len,
    ):
        # confirm all K and V keys match Q (i.e. MHA)
        assert_expected(num_heads, mha_parallel_attention.num_kv)
        # confirm num Q matches num_heads
        assert_expected(num_heads, mha_parallel_attention.num_heads)

        # input_ones = torch.ones(dims, dtype=torch.float)

        x = torch.randint(0, 256, (1, max_seq_len, embedding_dim))  # bs =1,
        attn_output = mha_parallel_attention(x)

        fixed_result_firstrow = torch.tensor(
            [
                15.9989,
                119.0005,
                32.0014,
                119.9999,
                113.9993,
                8.9996,
                141.0015,
                200.0015,
                136.9985,
                238.9991,
                236.0013,
                144.9993,
                224.9991,
                165.9994,
                193.9994,
                93.0001,
            ],
            dtype=torch.float32,
        )
        fixed_output_shape = torch.Size([1, max_seq_len, embedding_dim])

        assert_expected(fixed_result_firstrow, attn_output[0][0], rtol=0, atol=1e-4)
        assert_expected(fixed_output_shape, attn_output.shape)

    def test_mqa_parallel_attention(
        self,
        mqa_parallel_attention,
        num_heads,
        mqa_num_heads,
        max_seq_len,
        embedding_dim,
    ):
        # confirm all K and V keys match MQA num heads (i.e. MQA == 1)
        assert_expected(mqa_num_heads, mqa_parallel_attention.num_kv)
        # confirm num Q matches num_heads
        assert_expected(num_heads, mqa_parallel_attention.num_heads)

        x = torch.randint(0, 256, (1, max_seq_len, embedding_dim))
        attn_output = mqa_parallel_attention(x)

        fixed_result_firstrow = torch.tensor(
            [
                91.9992,
                24.0038,
                237.9937,
                74.0036,
                186.0031,
                53.0041,
                106.0050,
                179.9931,
                190.9989,
                178.9995,
                82.0005,
                190.9972,
                213.0028,
                213.9989,
                12.0008,
                190.9990,
            ],
            dtype=torch.float32,
        )
        fixed_output_shape = torch.Size([1, max_seq_len, embedding_dim])
        assert_expected(fixed_output_shape, attn_output.shape)
        # print(f"{attn_output[0][0]}")
        assert_expected(fixed_result_firstrow, attn_output[0][0], rtol=0, atol=1e-4)

    def test_gqa_parallel_attention(
        self,
        gqa_parallel_attention,
        num_heads,
        gqa_num_heads,
        max_seq_len,
        embedding_dim,
    ):
        # confirm all K and V keys match GQA num heads (i.e. GQA >= 2)
        assert_expected(gqa_num_heads, gqa_parallel_attention.num_kv)
        # confirm num Q matches num_heads
        assert_expected(num_heads, gqa_parallel_attention.num_heads)

        x = torch.randint(0, 256, (1, max_seq_len, embedding_dim))
        attn_output = gqa_parallel_attention(x)

        fixed_result_firstrow = torch.tensor(
            [
                201.0000,
                138.0011,
                196.9992,
                82.9997,
                4.9996,
                211.9985,
                103.9994,
                15.9996,
                177.0008,
                140.9993,
                213.9985,
                199.0000,
                146.9993,
                207.0003,
                109.0001,
                3.0005,
            ],
            dtype=torch.float32,
        )
        fixed_output_shape = torch.Size([1, max_seq_len, embedding_dim])
        assert_expected(fixed_output_shape, attn_output.shape)
        assert_expected(fixed_result_firstrow, attn_output[0][0], rtol=0, atol=1e-4)
