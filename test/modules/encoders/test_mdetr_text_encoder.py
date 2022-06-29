# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.modules.encoders.mdetr_text_encoder import (
    MDETRTextEmbeddings,
    MDETRTextEncoder,
    ModifiedTransformerEncoder,
)


class TestMDETRTextEncoder(unittest.TestCase):
    def setUp(self):
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
        set_rng_seed(0)

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
                -0.0304,
                -0.3717,
                0.5817,
                -0.8700,
                0.7169,
                -0.4643,
                0.8175,
                -0.4136,
                1.4704,
                -1.0958,
                1.4811,
                1.6897,
                0.8436,
                0.2235,
                1.5695,
                -0.7361,
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
                0.8221,
                0.8772,
                0.8121,
                0.9098,
                0.7683,
                1.1467,
                0.8986,
                0.9859,
                1.0459,
                0.8101,
                1.0133,
                0.8314,
                1.0375,
                0.9280,
                0.9604,
                0.8452,
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
                -1.6693,
                -1.5718,
                -1.4614,
                -1.6710,
                -1.5591,
                -1.5862,
                -1.3247,
                -1.6061,
                -1.5178,
                -1.6653,
                -1.4223,
                -1.4895,
                -1.5872,
                -1.4665,
                -1.5310,
                -1.7176,
            ]
        )
        out = self.text_encoder(self.input_ids, self.attention_mask)
        actual = out[1, :, 1]
        self.assertEqual(
            out.size(), (self.batch_size, self.input_length, self.hidden_size)
        )
        assert_expected(actual, expected, rtol=0.0, atol=1e-4)
