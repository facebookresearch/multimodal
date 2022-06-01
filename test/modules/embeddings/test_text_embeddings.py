# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.embeddings.text_embeddings import BERTEmbeddings


class TestBERTEmbeddings(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.embedding = BERTEmbeddings()
        self.input_ids = torch.randint(low=1, high=150, size=(5, 10)).long()
        self.segment_ids = torch.ones(5, 10).long()

    def test_text_embedding_shape(self):
        output = self.embedding(self.input_ids, self.segment_ids)
        assert_expected(output.shape, torch.Size([5, 10, 768]))

    def test_text_embedding_values(self):
        output = self.embedding(self.input_ids, self.segment_ids)

        actual_first_values = output[0, :, 0]
        actual_last_values = output[-1, :, -1]

        expected_first_values = torch.Tensor(
            [
                -0.5289,
                0.4380,
                -0.2358,
                0.1231,
                -0.3958,
                0.1124,
                0.3287,
                0.3287,
                0.6593,
                -0.6662,
            ]
        )

        expected_last_values = torch.Tensor(
            [
                0.6998,
                1.9490,
                0.8495,
                1.8604,
                1.0771,
                0.7184,
                1.5674,
                1.2117,
                0.0000,
                1.3389,
            ]
        )

        assert_expected(actual_first_values, expected_first_values, rtol=0.0, atol=1e-4)
        assert_expected(actual_last_values, expected_last_values, rtol=0.0, atol=1e-4)
