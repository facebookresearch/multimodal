# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from examples.cnn_lstm.cnn_lstm import cnn_lstm_classifier


class TestCNNLSTMModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.classifier_in_dim = 450
        self.num_classes = 32

    def test_forward(self):

        cnn_lstm = cnn_lstm_classifier(
            text_vocab_size=80,
            text_embedding_dim=20,
            cnn_input_dims=[3, 64, 128, 128, 64, 64],
            cnn_output_dims=[64, 128, 128, 64, 64, 10],
            cnn_kernel_sizes=[7, 5, 5, 5, 5, 1],
            lstm_input_size=20,
            lstm_hidden_dim=50,
            lstm_bidirectional=True,
            lstm_batch_first=True,
            classifier_in_dim=self.classifier_in_dim,
            num_classes=self.num_classes,
        )
        self.assertTrue(isinstance(cnn_lstm, torch.nn.Module))
        text = torch.randint(1, 79, (10,), dtype=torch.long).unsqueeze(0)
        image = torch.randn(3, 320, 480).unsqueeze(0)

        scores = cnn_lstm({"image": image, "text": text})
        self.assertEqual(scores.size(), torch.Size((1, 32)))
