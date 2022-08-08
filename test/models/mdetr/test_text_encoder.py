# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.models.mdetr.text_encoder import (
    MDETRTextEmbeddings,
    MDETRTextEncoder,
    ModifiedTransformerEncoder,
)


class TestMDETRTextEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.max_position_embeddings = 514
        self.hidden_size = 768
        self.embeddings = MDETRTextEmbeddings(
            hidden_size=self.hidden_size,
            vocab_size=50265,
            pad_token_id=1,
            type_vocab_size=1,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=1e-05,
            hidden_dropout_prob=0.1,
        )

        self.modified_transformer_encoder = ModifiedTransformerEncoder(
            embedding_dim=self.hidden_size,
            ffn_dimension=3072,
            num_attention_heads=12,
            num_encoder_layers=12,
            dropout=0.1,
            normalize_before=False,
        )

        self.text_encoder = MDETRTextEncoder(
            embeddings=self.embeddings, encoder=self.modified_transformer_encoder
        )
        self.text_encoder.eval()

        self.input_ids = torch.tensor(
            [
                [0, 100, 64, 192, 5, 3778, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    0,
                    1708,
                    190,
                    114,
                    38,
                    1395,
                    192,
                    5,
                    3778,
                    6,
                    38,
                    216,
                    14,
                    24,
                    8785,
                    2,
                ],
            ],
            dtype=torch.int,
        )
        self.attention_mask = torch.tensor(
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            ]
        )
        self.encoder_input = torch.rand((2, 16, 768))
        self.batch_size, self.input_length = self.input_ids.size()

    def test_mdetr_text_embeddings(self):
        expected = torch.Tensor(
            [
                -1.0921,
                2.2603,
                -0.5833,
                0.7053,
                0.2295,
                0.2110,
                -0.7579,
                -0.3196,
                0.1942,
                0.2076,
                -0.9220,
                0.0716,
                0.2924,
                0.2390,
                0.2598,
                1.3811,
            ]
        )
        out = self.embeddings(self.input_ids)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)

    def test_mdetr_modified_transformer(self):
        expected = torch.Tensor(
            [
                1.2321,
                0.9876,
                0.8055,
                0.9674,
                1.1693,
                1.0343,
                1.0212,
                1.0490,
                0.9856,
                1.1604,
                1.0352,
                0.9186,
                0.9872,
                1.0180,
                1.0587,
                1.0421,
            ]
        )
        out = self.modified_transformer_encoder(self.encoder_input, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)

    def test_mdetr_text_encoder(self):
        expected = torch.Tensor(
            [
                2.4597,
                2.6349,
                2.5019,
                2.3781,
                2.7154,
                2.5823,
                2.4751,
                2.5483,
                2.5868,
                2.5241,
                2.5561,
                2.6130,
                2.6505,
                2.3894,
                2.4084,
                2.7014,
            ]
        )
        out = self.text_encoder(self.input_ids, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)
