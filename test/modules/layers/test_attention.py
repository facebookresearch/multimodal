# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.attention import (
    AxialAttention,
    AxialAttentionBlock,
    FullAttention,
    merge_multihead,
    MultiHeadAttention,
    scaled_dot_product_attention,
    split_multihead,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


@pytest.fixture
def hidden_dim():
    return 3


@pytest.fixture
def n_dim():
    return 3


@pytest.fixture
def input_shape(n_dim):
    return tuple(repeat(2, n_dim))


@pytest.fixture
def q(input_shape, hidden_dim):
    n_heads = 1
    return torch.randn(1, n_heads, *input_shape, hidden_dim // n_heads)


@pytest.fixture
def kv(input_shape, hidden_dim):
    n_heads = 1
    return torch.randn(1, n_heads, *input_shape, hidden_dim // n_heads)


@pytest.fixture
def full_attn(input_shape):
    return FullAttention(input_shape, causal=False, attn_dropout=0.0)


@pytest.fixture
def axial_attn():
    return AxialAttention(1)  # only on second axis of input


class TestMultiheadAttention:
    @pytest.fixture
    def multihead_attn(self, input_shape, hidden_dim):
        def create_multihead_attn(n_heads, causal, attn_module):
            return MultiHeadAttention(
                input_shape, hidden_dim, hidden_dim, n_heads, 1, causal, attn_module
            )

        return create_multihead_attn

    def test_multi_head_attention(
        self,
        input_shape,
        hidden_dim,
        multihead_attn,
        full_attn,
    ):
        mha = multihead_attn(1, False, full_attn)
        q = 2 * torch.ones(1, *input_shape, hidden_dim)
        k = 2 * torch.ones(1, *input_shape, hidden_dim)
        v = 2 * torch.ones(1, *input_shape, hidden_dim)
        actual = mha(q, k, v)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                    ],
                    [
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                    ],
                ],
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_multi_head_attention_use_cache(
        self, input_shape, hidden_dim, multihead_attn, full_attn, mocker
    ):
        mha = multihead_attn(1, False, full_attn)
        mock_projection_k = mocker.patch.object(
            mha.w_ks, "forward", wraps=mha.w_ks.forward
        )
        mock_projection_v = mocker.patch.object(
            mha.w_vs, "forward", wraps=mha.w_vs.forward
        )

        q = 2 * torch.ones(1, *input_shape, hidden_dim)
        k = 2 * torch.ones(1, *input_shape, hidden_dim)
        v = 2 * torch.ones(1, *input_shape, hidden_dim)

        expected = torch.tensor(
            [
                [
                    [
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                    ],
                    [
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                        [[0.7469, 0.0560, 1.4119], [0.7469, 0.0560, 1.4119]],
                    ],
                ],
            ]
        )

        # cached k, v are linearly projected and split-headed: (b, n_heads, d1, ..., dn, emb_dim)
        expected_k = torch.tensor(
            [
                [
                    [
                        [
                            [[1.6964, -0.2340, 0.4197], [1.6964, -0.2340, 0.4197]],
                            [[1.6964, -0.2340, 0.4197], [1.6964, -0.2340, 0.4197]],
                        ],
                        [
                            [[1.6964, -0.2340, 0.4197], [1.6964, -0.2340, 0.4197]],
                            [[1.6964, -0.2340, 0.4197], [1.6964, -0.2340, 0.4197]],
                        ],
                    ],
                ],
            ]
        )
        expected_v = torch.tensor(
            [
                [
                    [
                        [
                            [[-1.6807, 0.7984, -0.4054], [-1.6807, 0.7984, -0.4054]],
                            [[-1.6807, 0.7984, -0.4054], [-1.6807, 0.7984, -0.4054]],
                        ],
                        [
                            [[-1.6807, 0.7984, -0.4054], [-1.6807, 0.7984, -0.4054]],
                            [[-1.6807, 0.7984, -0.4054], [-1.6807, 0.7984, -0.4054]],
                        ],
                    ],
                ],
            ]
        )

        # initially the cache should be empty
        assert not mha.cache
        for i in range(2):
            # pertube the input k, v but cache only once
            actual = mha(q, k + i, v + i, use_cache=True)
            assert_expected(mha.cache["k"], expected_k, rtol=0, atol=1e-4)
            assert_expected(mha.cache["v"], expected_v, rtol=0, atol=1e-4)
            assert_expected(actual, expected, rtol=0, atol=1e-4)
            # test that k, v projection is skipped except for the first pass
            mock_projection_k.assert_called_once()
            mock_projection_v.assert_called_once()

    def test_multi_head_attention_causal_use_cache(
        self, input_shape, hidden_dim, multihead_attn, full_attn
    ):
        n_heads = 1
        mha = multihead_attn(n_heads, True, full_attn)
        seq_len = torch.prod(torch.tensor(input_shape)).item()
        q = 2 * torch.ones(1, *input_shape, hidden_dim).flatten(start_dim=1, end_dim=-2)
        k = 2 * torch.ones(1, *input_shape, hidden_dim).flatten(start_dim=1, end_dim=-2)
        v = 2 * torch.ones(1, *input_shape, hidden_dim).flatten(start_dim=1, end_dim=-2)
        out = []
        # initially the cache should be empty
        assert not mha.cache
        # decoding is step-wise along the sequence dim
        for i in range(seq_len):
            out.append(
                mha(q[:, i : i + 1], k[:, i : i + 1], v[:, i : i + 1], use_cache=True)
            )
            # cached k, v are flattened and augmented by 1 unit at each step
            expected_kv_shape = torch.Size([1, n_heads, (i + 1), hidden_dim])
            assert_expected(mha.cache["k"].shape, expected_kv_shape)
            assert_expected(mha.cache["v"].shape, expected_kv_shape)

        out = torch.cat(out, dim=1)
        assert_expected(out.shape, torch.Size([1, seq_len, hidden_dim]))


def test_scaled_dot_product_attention(q, kv):
    actual, _ = scaled_dot_product_attention(q, kv, kv)
    expected = torch.tensor(
        [
            [
                [
                    [
                        [[-0.5862, 1.7955, 1.0711], [-0.2718, 1.2177, 1.4946]],
                        [[-0.0613, 0.1774, 0.4893], [0.6899, -0.0650, 0.2909]],
                    ],
                    [
                        [[0.2950, 1.2029, 1.7035], [0.2735, 0.5582, 0.6797]],
                        [[-1.1558, 1.0143, 0.1598], [0.7875, 0.0928, -0.7952]],
                    ],
                ],
            ],
        ]
    )
    assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_full_attention(full_attn, q, kv):
    k = v = kv
    actual = full_attn(q, k, v)
    # Output of full attention should be same as scaled_dot_product_attention
    # since input dims are flattened
    expected = torch.tensor(
        [
            [
                [
                    [
                        [[-0.4851, 1.2020, 0.7056], [0.3507, 0.3822, 0.2783]],
                        [[-0.8302, 1.1415, 0.4297], [-0.0969, 1.0956, 0.9591]],
                    ],
                    [
                        [[-0.0698, 0.9357, 1.4559], [-0.7157, 1.3919, 0.5880]],
                        [[-0.0598, 1.1194, 1.5332], [0.5494, -0.0489, -0.4454]],
                    ],
                ]
            ]
        ]
    )
    assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_axial_attention(axial_attn, q, kv):
    k = v = kv
    actual = axial_attn(q, k, v)
    expected = torch.tensor(
        [
            [
                [
                    [
                        [[-0.5869, 1.8958, 0.8688], [0.0299, 0.2098, 1.2741]],
                        [[-0.6662, 1.9747, 0.8980], [0.1002, 0.2094, 1.5472]],
                    ],
                    [
                        [[0.5902, -0.3275, -0.8727], [-1.0557, 1.0791, 0.3916]],
                        [[0.6623, -0.3223, -0.8948], [-1.0755, 1.0763, 0.3708]],
                    ],
                ]
            ]
        ]
    )
    assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_split_multihead(input_shape):
    x = torch.randn(1, *input_shape, 6)  # (b, d1, ..., dn, c)
    out = split_multihead(x, 2)
    actual = torch.tensor(out.shape)
    expected = torch.tensor((1, 2, *input_shape, 3))  # (b, h, d1, ..., dn, c // h)
    assert_expected(actual, expected)


def test_merge_multihead(input_shape, hidden_dim, q):
    out = merge_multihead(q)
    actual = torch.tensor(out.shape)
    expected = torch.tensor((1, *input_shape, hidden_dim))
    assert_expected(actual, expected)


class TestAxialBlock:
    @pytest.fixture
    def axial_block(self, input_shape, hidden_dim):
        return AxialAttentionBlock(len(input_shape), hidden_dim, 1)

    def test_axial_block_forward(self, axial_block, hidden_dim, input_shape):
        """Test AxialAttentionBlock with sub-components"""
        x = 2 * torch.ones(1, hidden_dim, *input_shape)
        actual = axial_block(x)
        expected = torch.tensor(
            [
                [
                    [
                        [[6.6573, 6.6573], [6.6573, 6.6573]],
                        [[6.6573, 6.6573], [6.6573, 6.6573]],
                    ],
                    [
                        [[4.1675, 4.1675], [4.1675, 4.1675]],
                        [[4.1675, 4.1675], [4.1675, 4.1675]],
                    ],
                    [
                        [[7.0048, 7.0048], [7.0048, 7.0048]],
                        [[7.0048, 7.0048], [7.0048, 7.0048]],
                    ],
                ],
            ]
        )

    def test_axial_block_channel_dim(self, axial_block, hidden_dim, input_shape):
        """Test dim check in forward of AxialAttentionBlock"""
        x = torch.zeros(1, hidden_dim + 1, *input_shape)
        with pytest.raises(ValueError):
            _ = axial_block(x)
