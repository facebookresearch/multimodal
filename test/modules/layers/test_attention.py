# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.attention import (
    AxialAttention,
    AxialAttentionBlock,
    FullAttention,
    MultiHeadAttention,
    scaled_dot_product_attention,
)


class TestAttention(unittest.TestCase):
    """
    Test all Attention classes
    """

    def setUp(self):
        set_rng_seed(4)
        self.hidden_dim = 3
        self.n_heads = 1
        self.input_shape = (2, 2, 2)
        self.q = torch.randn(
            1, self.n_heads, *self.input_shape, self.hidden_dim // self.n_heads
        )
        self.k = torch.randn(
            1, self.n_heads, *self.input_shape, self.hidden_dim // self.n_heads
        )
        self.v = torch.randn(
            1, self.n_heads, *self.input_shape, self.hidden_dim // self.n_heads
        )
        self.full = FullAttention(self.input_shape, causal=False, attn_dropout=0.0)
        self.ax = AxialAttention(1)  # only on second axis of input
        self.mha = MultiHeadAttention(
            self.input_shape,
            self.hidden_dim,
            self.hidden_dim,
            self.n_heads,
            1,
            causal=False,
            attn_module=self.full,
        )
        self.block = AxialAttentionBlock(
            len(self.input_shape), self.hidden_dim, self.n_heads
        )

    def test_scaled_dot_product_attention(self):
        actual = scaled_dot_product_attention(self.q, self.k, self.v)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [[0.7199, 2.2441, -0.7576], [0.4518, 1.5191, -0.2356]],
                            [[-1.1097, -0.1524, 0.3367], [0.0885, -0.2590, 0.4254]],
                        ],
                        [
                            [[-0.1849, 0.3928, 0.3666], [-0.5445, 0.0442, -0.0061]],
                            [[0.8435, -1.4510, -1.1567], [0.2037, -0.9690, -0.4564]],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_full_attention(self):
        actual = self.full(self.q, self.k, self.v)
        # Output of full attention should be same as scaled_dot_product_attention
        # since input dims are flattened
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [[0.4130, 0.5607, -0.6003], [0.1206, -0.0833, -0.1378]],
                            [[0.5494, -0.1801, -0.8837], [0.3011, 0.7369, -0.2519]],
                        ],
                        [
                            [[0.1344, 0.5524, 0.0436], [0.6117, 0.6719, -0.8588]],
                            [[0.1731, 0.8062, 0.0261], [-0.2240, -0.5229, -0.2820]],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_axial_attention(self):
        actual = self.ax(self.q, self.k, self.v)
        expected = torch.tensor(
            [
                [
                    [
                        [
                            [[0.8644, 2.3747, -0.8809], [-0.7204, 0.0344, 0.4795]],
                            [[0.8348, 2.4704, -0.9301], [-0.5203, 0.0964, 0.5355]],
                        ],
                        [
                            [[-0.7800, -0.5387, -0.4397], [0.7498, -1.2456, -0.9972]],
                            [[-0.7235, -0.5575, -0.4205], [0.7629, -1.2702, -1.0178]],
                        ],
                    ]
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_split_multihead(self):
        x = torch.randn(1, *self.input_shape, 6)
        self.mha.n_head = 2
        out = self.mha._split_multihead(x)
        actual = torch.tensor(out.shape)
        expected = torch.tensor((1, 2, *self.input_shape, 3))
        assert_expected(actual, expected)

    def test_combine_multihead(self):
        out = self.mha._combine_multihead(self.q)
        actual = torch.tensor(out.shape)
        expected = torch.tensor((1, *self.input_shape, self.hidden_dim))
        assert_expected(actual, expected)

    def test_multi_head_attention(self):
        # New tensors because need unflattened shape
        q = 2 * torch.ones(1, *self.input_shape, self.hidden_dim)
        k = 2 * torch.ones(1, *self.input_shape, self.hidden_dim)
        v = 2 * torch.ones(1, *self.input_shape, self.hidden_dim)
        actual = self.mha(q, k, v)
        expected = torch.tensor(
            [
                [
                    [
                        [[2.4187, 4.1634, 1.4579], [2.4187, 4.1634, 1.4579]],
                        [[2.4187, 4.1634, 1.4579], [2.4187, 4.1634, 1.4579]],
                    ],
                    [
                        [[2.4187, 4.1634, 1.4579], [2.4187, 4.1634, 1.4579]],
                        [[2.4187, 4.1634, 1.4579], [2.4187, 4.1634, 1.4579]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_axial_block_mha_length(self):
        """Test AxialAttentionBlock number of MHAs"""
        assert len(self.block.mha_attns) == 3, "incorrect number of MHAs"

    def test_axial_block_forward(self):
        """Test AxialAttentionBlock with sub-components"""
        x = 2 * torch.ones(1, self.hidden_dim, *self.input_shape)
        actual = self.block(x)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.5137, -0.5137], [-0.5137, -0.5137]],
                        [[-0.5137, -0.5137], [-0.5137, -0.5137]],
                    ],
                    [
                        [[1.7030, 1.7030], [1.7030, 1.7030]],
                        [[1.7030, 1.7030], [1.7030, 1.7030]],
                    ],
                    [
                        [[-5.2132, -5.2132], [-5.2132, -5.2132]],
                        [[-5.2132, -5.2132], [-5.2132, -5.2132]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_axial_block_channel_dim(self):
        """Test dim check in forward of AxialAttentionBlock"""
        x = torch.zeros(1, self.hidden_dim + 1, *self.input_shape)
        with self.assertRaises(ValueError):
            _ = self.block(x)
