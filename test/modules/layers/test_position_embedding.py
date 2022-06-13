# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from test.test_utils import assert_expected
from torch import nn
from torchmultimodal.modules.layers.position_embedding import (
    BroadcastedPositionEmbedding,
)


@pytest.fixture(scope="class")
def pos_emb():
    return BroadcastedPositionEmbedding(
        shape=(1, 2),
        embedding_dim=6,
    )


class TestBroadcastedPositionEmbedding:
    def test_init_sets_embedding(self, pos_emb):
        """Test the embeddings are initialized with the correct dimensions"""
        expected = [(1, 3), (2, 3)]
        for i, (key, _) in enumerate(pos_emb.embedding.items()):
            assert_expected(pos_emb.embedding[key].shape, expected[i])

    def test_init_bad_embedding_dim(self):
        """Test raising error when the embedding dim is not allowed"""
        with pytest.raises(ValueError):
            BroadcastedPositionEmbedding(shape=(1, 2), embedding_dim=5)

    def test_seq_len(self, pos_emb):
        assert_expected(pos_emb.seq_len, 2)

    def test_broadcast(self, pos_emb):
        """Test embedding along each dim is broadcasted correctly"""
        embedding = [
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8]]),
        ]
        expected = [
            torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]]]),
            torch.tensor([[[[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]),
        ]
        for i, emb in enumerate(embedding):
            pos_emb.embedding[f"d_{i}"] = nn.Parameter(emb)
            assert_expected(pos_emb._broadcast(i), expected[i])

    def test_decode(self, pos_emb):
        """Test the embedding at a previous location is selected for each decode step"""
        x_shape = (1, 2, 6)
        broadcasted_embedding = torch.tensor(
            [[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]]
        )
        expected = [
            torch.tensor([[[[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]]]),
            torch.tensor([[[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]]]),
        ]

        for decode_step, _ in enumerate(pos_emb.decode_idxs):
            actual = pos_emb._decode(decode_step, broadcasted_embedding, x_shape)
            assert_expected(actual, expected[decode_step])

    def test_forward(self, pos_emb):
        expected = (1, 2, 6)
        assert_expected(pos_emb().shape, expected)

    def test_forward_decode(self, pos_emb):
        """Test the decode statement inside ``forward`` is hit when ``decode_step`` is given"""
        x = torch.zeros(1, *(pos_emb.shape), pos_emb.embedding_dim).flatten(
            start_dim=1, end_dim=-2
        )
        actual = pos_emb(x, decode_step=0).shape
        expected = (1, 1, 6)
        assert_expected(actual, expected)
