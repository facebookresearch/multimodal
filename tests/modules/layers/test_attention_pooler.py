# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed
from torchmultimodal.modules.layers.attention_pooler import (
    AttentionPooler,
    CascadedAttentionPooler,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestAttentionPooler:
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def input_embed_dim(self):
        return 4

    @pytest.fixture
    def output_embed_dim(self):
        return 6

    @pytest.fixture
    def cascaded_output_embed_dim(self):
        return 10

    @pytest.fixture
    def seq_len(self):
        return 8

    @pytest.fixture
    def n_head(self):
        return 2

    @pytest.fixture
    def n_queries(self):
        return 12

    @pytest.fixture
    def inputs(self, batch_size, seq_len, input_embed_dim):
        return torch.randn(batch_size, seq_len, input_embed_dim)

    @pytest.fixture
    def pooler(self, input_embed_dim, output_embed_dim, n_head, n_queries):
        pooler = AttentionPooler(
            input_embed_dim=input_embed_dim,
            output_embed_dim=output_embed_dim,
            n_head=n_head,
            n_queries=n_queries,
        )
        init_weights_with_constant(pooler)
        return pooler

    @pytest.fixture
    def cascaded_pooler(
        self, pooler, output_embed_dim, cascaded_output_embed_dim, n_head
    ):
        second_pooler = AttentionPooler(
            input_embed_dim=output_embed_dim,
            output_embed_dim=cascaded_output_embed_dim,
            n_head=n_head,
            n_queries=1,
        )
        init_weights_with_constant(second_pooler)
        cascaded_pooler = CascadedAttentionPooler([pooler, second_pooler])
        return cascaded_pooler

    def test_forward(self, pooler, inputs, batch_size, n_queries, output_embed_dim):
        actual = pooler(inputs)
        expected_shape = (batch_size, n_queries, output_embed_dim)
        expected_sum = torch.tensor(144.0)
        assert_expected(actual.shape, expected_shape)
        assert_expected(actual.sum(), expected_sum)

    def test_torchscript(self, pooler, inputs):
        scripted_pooler = torch.jit.script(pooler)
        out = pooler(inputs)
        scripted_out = scripted_pooler(inputs)
        assert_expected(scripted_out, out)

    def test_cascaded_pooler_forward(
        self,
        cascaded_pooler,
        inputs,
        batch_size,
        n_queries,
        output_embed_dim,
        cascaded_output_embed_dim,
    ):
        actual = cascaded_pooler(inputs)

        expected_shapes = [
            (batch_size, n_queries, output_embed_dim),
            (batch_size, 1, cascaded_output_embed_dim),
        ]
        expected_sums = [torch.tensor(144.0), torch.tensor(20.0)]
        assert_expected([x.shape for x in actual], expected_shapes)
        assert_expected([x.sum() for x in actual], expected_sums)

    def test_cascaded_pooler_torchscript(self, cascaded_pooler, inputs):
        scripted_pooler = torch.jit.script(cascaded_pooler)
        out = cascaded_pooler(inputs)
        scripted_out = scripted_pooler(inputs)
        assert_expected(scripted_out, out)
