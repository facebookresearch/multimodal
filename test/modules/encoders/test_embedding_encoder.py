# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torchmultimodal.modules.encoders.embedding_encoder import EmbeddingEncoder


class TestEmbeddingEncoder(unittest.TestCase):
    def setUp(self):
        self.num_embeddings = 4
        self.embedding_dim = 4
        self.batch_size = 2
        self.input_size = 6
        embedding_weight = torch.Tensor([[4, 3, 2, 5], [2, 5, 6, 7], [1, 2, 0, 1]])
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.data = torch.LongTensor([[1, 2], [0, 1]])

    def test_embedding_encoder_sum(self):
        encoder = EmbeddingEncoder(self.embedding, "sum")
        actual = encoder(self.data)
        expected = torch.FloatTensor([[3, 7, 6, 8], [6, 8, 8, 12]])
        assert_expected(actual, expected)

    def test_embedding_encoder_mean(self):
        encoder = EmbeddingEncoder(self.embedding, "mean")
        actual = encoder(self.data)
        expected = torch.FloatTensor([[1.5, 3.5, 3, 4], [3, 4, 4, 6]])
        assert_expected(actual, expected)

    def test_embedding_encoder_max(self):
        encoder = EmbeddingEncoder(self.embedding, "max")
        actual = encoder(self.data)
        expected = torch.FloatTensor([[2, 5, 6, 7], [4, 5, 6, 7]])
        assert_expected(actual, expected)

    def test_embedding_encoder_hash(self):
        encoder = EmbeddingEncoder(self.embedding, "sum", use_hash=True)
        data = torch.LongTensor([[1, 2], [7, 9]])
        actual = encoder(data)
        expected = torch.FloatTensor([[3, 7, 6, 8], [2, 4, 0, 2]])
        assert_expected(actual, expected)

    def test_embedding_encoder_invalid_pooling(self):
        with self.assertRaises(ValueError):
            EmbeddingEncoder(self.embedding, "random")
