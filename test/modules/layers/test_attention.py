# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.layers.attention import (
    AxialAttention,
    BroadcastPositionEmbedding,
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


@pytest.fixture(scope="class")
def build_pos_emb():
    def _build_pos_emb(dim, shape=torch.Size([1, 2]), embedding_dim=6):
        return BroadcastPositionEmbedding(
            shape=shape,
            embedding_dim=embedding_dim,
            dim=dim,
        )

    return _build_pos_emb


class TestBroadcastPositionEmbedding:

    TEST_BROADCAST_DATA = [
        (
            -1,
            [
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8]]),
            ],
            [
                torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]]),
                torch.tensor([[[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]),
            ],
        ),
        (
            1,
            [
                torch.tensor([[0.0], [1.0], [2.0]]),
                torch.tensor([[3.0, 6.0], [4.0, 7.0], [5.0, 8.0]]),
            ],
            [
                torch.tensor([[[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]]),
                torch.tensor([[[[3.0, 6.0]], [[4.0, 7.0]], [[5.0, 8.0]]]]),
            ],
        ),
    ]

    TEST_DECODE_DATA = [
        (
            -1,
            torch.tensor([1, 2, 6]),
            torch.tensor(
                [[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]]
            ),
            [
                torch.tensor([[[[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]]),
                torch.tensor([[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]]),
            ],
        ),
        (
            1,
            torch.tensor([1, 6, 2]),
            torch.tensor(
                [
                    [
                        [[0.0, 7.0]],
                        [[1.0, 8.0]],
                        [[2.0, 9.0]],
                        [[3.0, 10.0]],
                        [[4.0, 11.0]],
                        [[5.0, 12.0]],
                    ]
                ]
            ),
            [
                torch.tensor(
                    [[[[7.0]], [[8.0]], [[9.0]], [[10.0]], [[11.0]], [[12.0]]]]
                ),
                torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]]),
            ],
        ),
    ]

    @pytest.mark.parametrize(
        "dim, expected",
        [
            (-1, [torch.Size([1, 3]), torch.Size([2, 3])]),
            (1, [torch.Size([3, 1]), torch.Size([3, 2])]),
        ],
    )
    def test_init_sets_embedding(self, dim, expected, build_pos_emb):
        """Test the embeddings are initialized with the correct dimensions"""
        pos_emb = build_pos_emb(dim)
        for i, (key, _) in enumerate(pos_emb.embedding.items()):
            assert_expected(pos_emb.embedding[key].shape, expected[i])

    def test_init_bad_dim(self, build_pos_emb):
        """Test raising error when the embedding is given at an illegal location"""
        with pytest.raises(ValueError):
            build_pos_emb(dim=-2)

    def test_init_bad_embedding_dim(self, build_pos_emb):
        """Test raising error when the embedding dim is not allowed"""
        with pytest.raises(ValueError):
            build_pos_emb(dim=-1, embedding_dim=5)

    def test_seq_len(self, build_pos_emb):
        pos_emb = build_pos_emb(-1)
        assert_expected(pos_emb.seq_len, 2)

    @pytest.mark.parametrize("dim, embedding, expected", TEST_BROADCAST_DATA)
    def test_broadcast(self, dim, embedding, expected, build_pos_emb):
        """Test embedding along each dim is broadcasted correctly"""
        pos_emb = build_pos_emb(dim)
        for i, emb in enumerate(embedding):
            assert_expected(pos_emb._broadcast(emb, i), expected[i])

    @pytest.mark.parametrize(
        "dim, x_shape, embedding_broadcast, expected", TEST_DECODE_DATA
    )
    def test_decode(self, dim, x_shape, embedding_broadcast, expected, build_pos_emb):
        """Test the embedding at a previous location is selected for each decode step"""
        pos_emb = build_pos_emb(dim)
        for decode_step, _ in enumerate(pos_emb.decode_idxs):
            actual = pos_emb._decode(decode_step, embedding_broadcast, x_shape)
            assert_expected(actual, expected[decode_step])

    @pytest.mark.parametrize(
        "dim, expected", [(-1, torch.Size([1, 2, 6])), (1, torch.Size([1, 6, 2]))]
    )
    def test_forward(self, dim, expected, build_pos_emb):
        pos_emb = build_pos_emb(dim)
        assert_expected(pos_emb().shape, expected)

    def test_forward_decode(self, build_pos_emb):
        """Test the decode statement inside ``forward`` is hit when ``decode_step`` is given"""
        pos_emb = build_pos_emb(dim=-1)
        x = torch.zeros(1, *(pos_emb.shape), pos_emb.embedding_dim).flatten(
            start_dim=1, end_dim=-2
        )
        actual = pos_emb(x, decode_step=0).shape
        expected = torch.Size([1, 1, 6])
        assert_expected(actual, expected)
