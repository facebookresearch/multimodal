# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.modules.layers.transformer_decoder import (
    RightShift,
    SiLU,
    #     TransformerDecoder,
    #     TransformerLayer,
)


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(4)


class DummyPositionEmbedding(nn.Module):
    def __init__(self, seq_len, embedding_dim):
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.embeddings = torch.ones(
            1, seq_len, embedding_dim
        )  # 1 is the batch dim to be broadcasted with data

    @property
    def decode_idxs(self):
        return list(torch.arange(self.seq_len))

    def forward(self, x, decode_step=None):
        if decode_step is not None:
            embeddings = self.embeddings[:, 1, :]

        return embeddings


def test_sigmoid_linear_unit():
    silu = SiLU()
    actual = silu(torch.ones(3))
    expected = torch.tensor([0.8458, 0.8458, 0.8458])
    assert_expected(actual, expected)


class TestRightShift:
    TEST_DATA_DECODE = [
        (
            0,
            torch.tensor(
                [
                    [
                        [-0.0321, 0.0046, 0.0448, 0.0169],
                        [1.0000, 1.0000, 1.0000, 1.0000],
                        [1.0000, 1.0000, 1.0000, 1.0000],
                    ]
                ]
            ),
        ),
        (
            1,
            torch.tensor(
                [
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ],
                ]
            ),
        ),
    ]

    @pytest.fixture
    def right_shift(self):
        embedding_dim = 4
        return RightShift(embedding_dim)

    def test_right_shift(self, right_shift):
        x = torch.ones(1, 3, 4)  # (batch, seq_len, embedding_dim)
        actual = right_shift(x)
        expected = torch.tensor(
            [
                [
                    [-0.0321, 0.0046, 0.0448, 0.0169],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                    [1.0000, 1.0000, 1.0000, 1.0000],
                ]
            ]
        )
        assert_expected(actual, expected, rtol=1e-5, atol=1e-4)


class TestTransformerLayer:
    def test_forward_training(self):
        pass

    def test_forward(self):
        pass


class TestTransformerDecoder:
    def test_get_training_embeddings(self):
        pass

    def test_get_inference_embeddings(self):
        pass

    def test_forward(self):
        pass

    def test_forward_decode(self):
        pass
