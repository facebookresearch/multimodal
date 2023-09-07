# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.position_embedding import (
    AlibiPositionEmbeddings,
    BroadcastedPositionEmbedding,
    SinusoidalPositionEmbeddings,
)


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(2023)


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


class TestAlibiPositionEmbedding:
    @pytest.fixture
    def max_seq_len(self):
        return 16

    @pytest.fixture
    def embedding_dim(self):
        return 32

    @pytest.fixture
    def num_heads(self):
        return 8

    def test_alibi_mask(
        self,
        max_seq_len,
        num_heads,
    ):
        alibi_class = AlibiPositionEmbeddings(
            max_seq_len=max_seq_len, num_heads=num_heads
        )
        base_mask = alibi_class.get_attention_mask(max_seq_len)

        # verify mask shape
        expected_shape = torch.Size((num_heads, max_seq_len, max_seq_len))
        assert_expected(base_mask.shape, expected_shape)

        # verify alibi mask components
        expected_last_head_row = torch.tensor(
            [
                -0.0586,
                -0.0547,
                -0.0508,
                -0.0469,
                -0.0430,
                -0.0391,
                -0.0352,
                -0.0312,
                -0.0273,
                -0.0234,
                -0.0195,
                -0.0156,
                -0.0117,
                -0.0078,
                -0.0039,
                0.0000,
            ]
        )

        expected_first_head_first_row_first_entry = torch.tensor(
            0.0000,
        )

        assert_expected(
            base_mask[0][0][0],
            expected_first_head_first_row_first_entry,
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            base_mask[num_heads - 1][max_seq_len - 1],
            expected_last_head_row,
            rtol=0,
            atol=1e-4,
        )
