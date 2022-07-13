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


class TestBroadcastedPositionEmbedding:
    @pytest.fixture(scope="class")
    def pos_emb(self):
        _pos_emb = BroadcastedPositionEmbedding(
            shape=(1, 2, 3),
            embedding_dim=6,
        )
        _pos_emb.embedding = nn.ParameterDict(
            {
                "d_0": nn.Parameter(torch.tensor([[0.0, 1.0]])),
                "d_1": nn.Parameter(torch.tensor([[2.0, 3.0], [4.0, 5.0]])),
                "d_2": nn.Parameter(torch.tensor([[6.0, 7.0], [8.0, 9.0], [0.0, 1.0]])),
            }
        )

        return _pos_emb

    def test_init_sets_embedding(self, pos_emb):
        """Test the embeddings are initialized with the correct dimensions"""
        expected = [(1, 2), (2, 2), (3, 2)]
        for i, (key, _) in enumerate(pos_emb.embedding.items()):
            assert_expected(pos_emb.embedding[key].shape, expected[i])

    def test_init_bad_embedding_dim(self):
        """Test raising error when the embedding dim is not allowed"""
        with pytest.raises(ValueError):
            BroadcastedPositionEmbedding(shape=(1, 2, 3), embedding_dim=5)

    def test_broadcast(self, pos_emb):
        """Test embedding along each dim is broadcasted correctly"""
        expected = [
            torch.tensor(
                [
                    [
                        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]],
                        [[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]],
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        [[6.0, 7.0], [8.0, 9.0], [0.0, 1.0]],
                        [[6.0, 7.0], [8.0, 9.0], [0.0, 1.0]],
                    ],
                ]
            ),
        ]
        for i in range(pos_emb.n_dim):
            assert_expected(pos_emb._broadcast(i), expected[i])

    def test_forward(self, pos_emb):
        """Test the correct embeddings are returned for the given position ids"""
        position_ids = torch.tensor([[1, 3, -1]])
        actual = pos_emb(position_ids)
        expected = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 8.0, 9.0],
                    [0.0, 1.0, 4.0, 5.0, 6.0, 7.0],
                    [0.0, 1.0, 4.0, 5.0, 0.0, 1.0],
                ]
            ]
        )
        assert_expected(actual, expected)

    def test_forward_invalid_input(self, pos_emb):
        """Test raising error when position ids contain illegal values"""
        with pytest.raises(IndexError):
            pos_emb(position_ids=torch.tensor([-2, 0]))
        with pytest.raises(IndexError):
            pos_emb(position_ids=torch.tensor([0, 6]))
