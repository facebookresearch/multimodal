# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import torch
from test.test_utils import assert_expected
from torchmultimodal.modules.layers.attention import BroadcastPositionEmbedding


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
