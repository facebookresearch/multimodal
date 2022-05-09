# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy

import torch
from test.test_utils import assert_tensors_equal
from torch import nn
from torchmultimodal.modules.encoders.weighted_embedding_encoder import (
    WeightedEmbeddingEncoder,
)


class TestEmbeddingEncoder(unittest.TestCase):
    def setUp(self):
        embedding_weights = torch.Tensor(
            [
                [1, 1],
                [
                    2,
                    2,
                ],
                [1, 0],
            ]
        )
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)

    def test_forward_sum_pooling(self):
        input = torch.Tensor(
            [
                [1, 0, 0.25, 0.75],
                [0, 1, 0.6, 0.4],
            ]
        )
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=self.embedding, pooling_function=torch.sum
        )
        actual = weighted_embedding_encoder(input)
        expected = torch.Tensor(
            [
                [1.25, 1.25],
                [1.4, 1.4],
            ]
        )
        assert_tensors_equal(actual, expected)

    def test_forward_mean_pooling(self):
        input = torch.Tensor(
            [
                [1, 0, 0.25, 0.75],
                [0, 1, 0.6, 0.4],
            ]
        )
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=self.embedding, pooling_function=torch.mean
        )
        actual = weighted_embedding_encoder(input)
        expected = torch.Tensor(
            [
                [0.625, 0.625],
                [0.7, 0.7],
            ]
        )
        assert_tensors_equal(actual, expected)

    def test_forward_max_pooling(self):
        input = torch.Tensor(
            [
                [1, 0, 0.25, 0.75],
                [0, 1, 0.6, 0.4],
            ]
        )
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=self.embedding, pooling_function=torch.max
        )
        actual = weighted_embedding_encoder(input)
        expected = torch.Tensor(
            [
                [0.75, 0.75],
                [0.8, 0.8],
            ]
        )
        assert_tensors_equal(actual, expected)

    def test_forward_hash_no_padding(self):
        input = torch.Tensor(
            [
                [1, 3, 0.25, 0.75],
                [6, 1, 0.6, 0.4],
            ]
        )
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=self.embedding, pooling_function=torch.max, use_hash=True
        )
        actual = weighted_embedding_encoder(input)
        expected = torch.Tensor(
            [
                [0.75, 0.75],
                [0.8, 0.8],
            ]
        )
        assert_tensors_equal(actual, expected)

    def test_forward_hash_zero_padding(self):
        input = torch.Tensor(
            [
                [0, 1, 0.25, 0.75],
                [6, 1, 0.6, 0.4],
            ]
        )
        embedding = deepcopy(self.embedding)
        embedding.padding_idx = 0
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=embedding, pooling_function=torch.sum, use_hash=True
        )
        actual = weighted_embedding_encoder(input)
        expected = torch.Tensor(
            [
                [1.75, 1.75],
                [1.4, 0.8],
            ]
        )
        assert_tensors_equal(actual, expected)

    def test_forward_hash_invalid_padding(self):
        embedding = deepcopy(self.embedding)
        embedding.padding_idx = 2
        self.assertRaises(
            ValueError, WeightedEmbeddingEncoder, embedding, torch.sum, 1, True
        )

    def test_scripting(self):
        input = torch.Tensor(
            [
                [1, 0, 0.25, 0.75],
                [0, 1, 0.6, 0.4],
            ]
        )
        weighted_embedding_encoder = WeightedEmbeddingEncoder(
            embedding=self.embedding, pooling_function=torch.mean, use_hash=True
        )
        scripted_encoder = torch.jit.script(weighted_embedding_encoder)
        actual = scripted_encoder(input)
        expected = torch.Tensor(
            [
                [0.625, 0.625],
                [0.7, 0.7],
            ]
        )
        assert_tensors_equal(actual, expected)
