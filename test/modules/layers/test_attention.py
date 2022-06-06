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
        self.input_shape = torch.tensor((2, 2, 2))
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
        self.ax = AxialAttention(3, 2)  # only on third axis of input
        self.mha = MultiHeadAttention(
            self.input_shape,
            self.hidden_dim,
            self.hidden_dim,
            self.n_heads,
            1,
            causal=False,
            attn_module=self.full,
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
        q = torch.randn(1, *self.input_shape, self.hidden_dim)
        k = torch.randn(1, *self.input_shape, self.hidden_dim)
        v = torch.randn(1, *self.input_shape, self.hidden_dim)
        actual = self.mha(q, k, v)
        expected = torch.tensor(
            [
                [
                    [
                        [[-0.1824, 0.2826, 0.4706], [-0.1540, 0.2962, 0.4301]],
                        [[-0.1795, 0.2889, 0.4178], [-1.2837, -0.2228, -0.6794]],
                    ],
                    [
                        [[-0.5227, 0.1744, 0.3691], [-0.3784, 0.2148, 0.3581]],
                        [[-1.0747, -0.1513, -0.4717], [-1.3936, -0.2522, -0.7915]],
                    ],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)
