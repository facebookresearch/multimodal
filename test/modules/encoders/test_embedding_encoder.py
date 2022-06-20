# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected
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
        embedding = encoder(self.data)
        self.assertEqual(embedding.size(), (self.batch_size, self.embedding_dim))
        self.assertTrue(
            torch.equal(embedding, torch.FloatTensor([[3, 7, 6, 8], [6, 8, 8, 12]]))
        )

    def test_embedding_encoder_mean(self):
        encoder = EmbeddingEncoder(self.embedding, "mean")
        embedding = encoder(self.data)
        self.assertEqual(embedding.size(), (self.batch_size, self.embedding_dim))
        self.assertTrue(
            torch.equal(embedding, torch.FloatTensor([[1.5, 3.5, 3, 4], [3, 4, 4, 6]]))
        )

    def test_embedding_encoder_max(self):
        encoder = EmbeddingEncoder(self.embedding, "max")
        embedding = encoder(self.data)
        self.assertEqual(embedding.size(), (self.batch_size, self.embedding_dim))
        self.assertTrue(
            torch.equal(embedding, torch.FloatTensor([[2, 5, 6, 7], [4, 5, 6, 7]]))
        )

    def test_embedding_encoder_hash(self):
        encoder = EmbeddingEncoder(self.embedding, "sum", use_hash=True)
        data = torch.LongTensor([[1, 2], [7, 9]])
        embedding = encoder(data)
        self.assertEqual(embedding.size(), (self.batch_size, self.embedding_dim))
        self.assertTrue(
            torch.equal(embedding, torch.FloatTensor([[3, 7, 6, 8], [2, 4, 0, 2]]))
        )

    def test_embedding_encoder_invalid_pooling(self):
        with self.assertRaises(ValueError):
            EmbeddingEncoder(self.embedding, "random")
