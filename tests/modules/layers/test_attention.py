# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import repeat

import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.attention import (
    merge_multihead,
    MultiHeadAttention,
    scaled_dot_product_attention,
    SelfAttention,
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
def self_attn():
    return SelfAttention(attn_dropout=0.0)


class TestMultiheadAttention:
    @pytest.fixture
    def multihead_attn(self, hidden_dim):
        def create_multihead_attn(n_heads, attn_module):
            return MultiHeadAttention(hidden_dim, hidden_dim, n_heads, attn_module)

        return create_multihead_attn

    def test_multi_head_self_attention(
        self,
        input_shape,
        hidden_dim,
        multihead_attn,
        self_attn,
    ):
        mha = multihead_attn(1, self_attn)
        qkv = 2 * torch.ones(1, *input_shape, hidden_dim)
        actual = mha(qkv)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [1.069666, 1.304498, -0.016060],
                            [1.069666, 1.304498, -0.016060],
                        ],
                        [
                            [1.069666, 1.304498, -0.016060],
                            [1.069666, 1.304498, -0.016060],
                        ],
                    ],
                    [
                        [
                            [1.069666, 1.304498, -0.016060],
                            [1.069666, 1.304498, -0.016060],
                        ],
                        [
                            [1.069666, 1.304498, -0.016060],
                            [1.069666, 1.304498, -0.016060],
                        ],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_multi_head_cross_attention(
        self,
        input_shape,
        hidden_dim,
        multihead_attn,
        self_attn,
    ):
        mha = multihead_attn(1, self_attn)
        q = 2 * torch.ones(1, *input_shape, hidden_dim)
        kv = torch.ones(1, *input_shape, hidden_dim)
        actual = mha(q, kv)
        expected = torch.tensor(
            [
                [
                    [
                        [[0.7675, 0.8126, -0.1126], [0.7675, 0.8126, -0.1126]],
                        [[0.7675, 0.8126, -0.1126], [0.7675, 0.8126, -0.1126]],
                    ],
                    [
                        [[0.7675, 0.8126, -0.1126], [0.7675, 0.8126, -0.1126]],
                        [[0.7675, 0.8126, -0.1126], [0.7675, 0.8126, -0.1126]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_multi_head_attention_use_cache(
        self, input_shape, hidden_dim, multihead_attn, self_attn, mocker
    ):
        mha = multihead_attn(1, self_attn)
        mock_projection_k = mocker.patch.object(
            mha.key, "forward", wraps=mha.key.forward
        )
        mock_projection_v = mocker.patch.object(
            mha.value, "forward", wraps=mha.value.forward
        )

        q = 2 * torch.ones(1, *input_shape, hidden_dim)
        kv = 2 * torch.ones(1, *input_shape, hidden_dim)

        expected = torch.tensor(
            [
                [
                    [
                        [[1.0697, 1.3045, -0.0161], [1.0697, 1.3045, -0.0161]],
                        [[1.0697, 1.3045, -0.0161], [1.0697, 1.3045, -0.0161]],
                    ],
                    [
                        [[1.0697, 1.3045, -0.0161], [1.0697, 1.3045, -0.0161]],
                        [[1.0697, 1.3045, -0.0161], [1.0697, 1.3045, -0.0161]],
                    ],
                ]
            ]
        )

        # cached k, v are linearly projected and split-headed: (b, n_heads, d1, ..., dn, emb_dim)
        expected_k = torch.tensor(
            [
                [
                    [
                        [
                            [
                                [0.935526, 0.753922, -1.434496],
                                [0.935526, 0.753922, -1.434496],
                            ],
                            [
                                [0.935526, 0.753922, -1.434496],
                                [0.935526, 0.753922, -1.434496],
                            ],
                        ],
                        [
                            [
                                [0.935526, 0.753922, -1.434496],
                                [0.935526, 0.753922, -1.434496],
                            ],
                            [
                                [0.935526, 0.753922, -1.434496],
                                [0.935526, 0.753922, -1.434496],
                            ],
                        ],
                    ]
                ]
            ]
        )
        expected_v = torch.tensor(
            [
                [
                    [
                        [
                            [[2.0164, 1.4426, 1.0050], [2.0164, 1.4426, 1.0050]],
                            [[2.0164, 1.4426, 1.0050], [2.0164, 1.4426, 1.0050]],
                        ],
                        [
                            [[2.0164, 1.4426, 1.0050], [2.0164, 1.4426, 1.0050]],
                            [[2.0164, 1.4426, 1.0050], [2.0164, 1.4426, 1.0050]],
                        ],
                    ]
                ]
            ]
        )

        # initially the cache should be empty
        assert not mha.cache
        for i in range(2):
            # pertube the input k, v but cache only once
            actual = mha(q, kv + i, use_cache=True)
            assert_expected(mha.cache["k"], expected_k, rtol=0, atol=1e-4)
            assert_expected(mha.cache["v"], expected_v, rtol=0, atol=1e-4)
            assert_expected(actual, expected, rtol=0, atol=1e-4)
            # test that k, v projection is skipped except for the first pass
            mock_projection_k.assert_called_once()
            mock_projection_v.assert_called_once()

    def test_multi_head_attention_causal_use_cache(
        self, input_shape, hidden_dim, multihead_attn, self_attn
    ):
        n_heads = 1
        mha = multihead_attn(n_heads, self_attn)
        seq_len = torch.prod(torch.tensor(input_shape)).item()
        q = 2 * torch.ones(1, *input_shape, hidden_dim).flatten(start_dim=1, end_dim=-2)
        kv = 2 * torch.ones(1, *input_shape, hidden_dim).flatten(
            start_dim=1, end_dim=-2
        )
        out = []
        # initially the cache should be empty
        assert not mha.cache
        # decoding is step-wise along the sequence dim
        for i in range(seq_len):
            out.append(
                mha(q[:, i : i + 1], kv[:, i : i + 1], use_cache=True, causal=True)
            )
            # cached k, v are flattened and augmented by 1 unit at each step
            expected_kv_shape = torch.Size([1, n_heads, (i + 1), hidden_dim])
            assert_expected(mha.cache["k"].shape, expected_kv_shape)
            assert_expected(mha.cache["v"].shape, expected_kv_shape)

        out = torch.cat(out, dim=1)
        assert_expected(out.shape, torch.Size([1, seq_len, hidden_dim]))


class TestScaledDotProductAttention:
    def test_scaled_dot_product_attention(self, q, kv):
        output = scaled_dot_product_attention(q, kv, kv)
        actual = output
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

    def test_scaled_dot_product_attention_with_attention_mask(self, q, kv):
        attn_shape = torch.Size([1, 1, 2, 2, 2, 2])
        mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]).view(
            attn_shape
        )
        actual = scaled_dot_product_attention(q, kv, kv, attention_mask=mask)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [[-0.7042, 2.0126, 0.9120], [-0.2718, 1.2177, 1.4946]],
                            [[-0.1652, 0.2109, 0.5167], [1.7146, -0.3956, 0.0204]],
                        ],
                        [
                            [[0.2950, 1.2029, 1.7035], [0.2973, 1.2710, 1.8117]],
                            [[1.5320, -0.2602, -1.1611], [0.7875, 0.0928, -0.7952]],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scaled_dot_product_attention_with_dropout(self, q, kv):
        actual = scaled_dot_product_attention(q, kv, kv, attn_dropout=0.3)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [
                                [0.0000e00, 0.0000e00, 0.0000e00],
                                [-5.6284e-01, 1.6085e00, 7.2891e-01],
                            ],
                            [
                                [1.3536e-01, -3.1232e-02, 1.6106e-03],
                                [9.8563e-01, -9.2847e-02, 4.1562e-01],
                            ],
                        ],
                        [
                            [
                                [4.2149e-01, 1.7184e00, 2.4336e00],
                                [2.3824e-01, 1.0184e00, 1.4517e00],
                            ],
                            [
                                [-1.6511e00, 1.4490e00, 2.2828e-01],
                                [1.1250e00, 1.3256e-01, -1.1361e00],
                            ],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)


def test_self_attention(self_attn, q, kv):
    k = v = kv
    actual = self_attn(q, k, v)
    # Output of self attention should be same as scaled_dot_product_attention
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


def test_sdpa_numerical_equivalence_with_manual(q, kv):
    """Verify that F.scaled_dot_product_attention (MATH backend) produces
    numerically identical results to the manual matmul->scale->mask->softmax->matmul
    implementation."""
    # Manual reference implementation
    attn = torch.matmul(q, kv.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    manual_out = torch.matmul(attn, kv)

    # Our implementation (uses F.scaled_dot_product_attention with MATH backend)
    sdpa_out = scaled_dot_product_attention(q, kv, kv)

    assert torch.allclose(sdpa_out, manual_out, rtol=1e-5, atol=1e-5), (
        f"Max diff: {(sdpa_out - manual_out).abs().max().item()}"
    )


def test_sdpa_numerical_equivalence_with_mask(q, kv):
    """Verify numerical equivalence when using an attention mask."""
    attn_shape = torch.Size([1, 1, 2, 2, 2, 2])
    mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]).view(
        attn_shape
    )

    # Manual reference implementation
    attn = torch.matmul(q, kv.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype))
    attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = torch.nn.functional.softmax(attn, dim=-1)
    manual_out = torch.matmul(attn, kv)

    # Our implementation
    sdpa_out = scaled_dot_product_attention(q, kv, kv, attention_mask=mask)

    assert torch.allclose(sdpa_out, manual_out, rtol=1e-5, atol=1e-5), (
        f"Max diff: {(sdpa_out - manual_out).abs().max().item()}"
    )
