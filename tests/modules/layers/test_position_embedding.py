# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from tests.test_utils import assert_expected
from torch import nn

from torchmultimodal.modules.layers.position_embedding import (
    BroadcastedPositionEmbedding,
    RotaryPositionalEmbeddings,
    SinusoidalPositionEmbeddings,
)


class TestBroadcastedPositionEmbedding:
    @pytest.fixture(scope="class")
    def pos_emb(self):
        _pos_emb = BroadcastedPositionEmbedding(
            latent_shape=(1, 2, 3),
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
            BroadcastedPositionEmbedding(latent_shape=(1, 2, 3), embedding_dim=5)

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
        with pytest.raises(IndexError) as exc_info:
            pos_emb(position_ids=torch.tensor([[-2, 0]]))
        assert exc_info.value.args[0] == "Invalid position ids: tensor([-2])"
        with pytest.raises(IndexError) as exc_info:
            pos_emb(position_ids=torch.tensor([[0, 6]]))
        assert exc_info.value.args[0] == "Invalid position ids: tensor([6])"


class TestSinusoidalPositionEmbeddings:
    @pytest.fixture
    def data(self):
        return torch.Tensor([1, 2, 3])

    @pytest.fixture
    def emb(self):
        return SinusoidalPositionEmbeddings(5)

    def test_forward(self, data, emb):
        actual = emb(data)
        expected = torch.Size([3, 5])
        assert_expected(actual.shape, expected)


def test_rotary_embeddings_math():
    q = (
        torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    )  # b h s e

    k = 2 * torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(
        0
    )  # b h s e

    rotary_embeddings = RotaryPositionalEmbeddings(2, 2, 1)
    qr, kr = rotary_embeddings(q, k, 0)
    rot0 = torch.tensor([[math.cos(0), -math.sin(0)], [math.sin(0), math.cos(0)]])
    rot1 = torch.tensor([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]])

    assert_expected(torch.matmul(rot0, q[..., 0, :].squeeze()), qr[..., 0, :].squeeze())
    assert_expected(torch.matmul(rot1, q[..., 1, :].squeeze()), qr[..., 1, :].squeeze())
    assert_expected(torch.matmul(rot0, k[..., 0, :].squeeze()), kr[..., 0, :].squeeze())
    assert_expected(torch.matmul(rot1, k[..., 1, :].squeeze()), kr[..., 1, :].squeeze())


def test_rotary_embeddings_left_padding():
    q = torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
    k = 2 * torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
    rotary_embeddings = RotaryPositionalEmbeddings(16, 32)

    qr, kr = rotary_embeddings(q, k, 0)
    qr2, kr2 = rotary_embeddings(q, k, torch.tensor([0, 1]))

    assert_expected(qr[0], qr2[0])
    assert_expected(qr[0, :, 1], qr2[1, :, 0])

    assert_expected(kr[0], kr2[0])
    assert_expected(kr[0, :, 1], kr2[1, :, 0])
