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
                0.4997,
                -0.3613,
                -0.7117,
                0.5731,
                0.3360,
                0.5778,
                0.9751,
                0.6629,
                0.6125,
                -0.9168,
            ]
        )

        expected_last_values = torch.Tensor(
            [
                -0.1139,
                2.1699,
                1.3947,
                1.7582,
                0.5508,
                0.7900,
                1.7724,
                1.7282,
                0.0000,
                1.0629,
            ]
        )

        assert_expected(actual_first_values, expected_first_values, rtol=0.0, atol=1e-4)
        assert_expected(actual_last_values, expected_last_values, rtol=0.0, atol=1e-4)
