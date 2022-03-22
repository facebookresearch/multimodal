# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
from torch import nn
from torchmultimodal.models.cnn_lstm import CNNLSTM
from torchmultimodal.modules.layers.mlp import MLP


class TestCNNLSTMModule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.classifier_in_dim = 450
        self.num_classes = 32

    def test_forward(self):
        classifier = partial(
            MLP,
            in_dim=self.classifier_in_dim,
            out_dim=self.num_classes,
            activation=nn.ReLU,
            normalization=nn.BatchNorm1d,
        )
        cnn_lstm = CNNLSTM(
            text_vocab_size=80,
            text_embedding_dim=20,
            cnn_input_dims=[3, 64, 128, 128, 64, 64],
            cnn_output_dims=[64, 128, 128, 64, 64, 10],
            cnn_kernel_sizes=[7, 5, 5, 5, 5, 1],
            lstm_input_size=20,
            lstm_hidden_dim=50,
            lstm_bidirectional=True,
            lstm_batch_first=True,
            classifier=classifier,
            classifier_in_dim=self.classifier_in_dim,
            num_classes=self.num_classes,
        )
        self.assertTrue(isinstance(cnn_lstm, torch.nn.Module))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        text = torch.randint(1, 79, (10,), dtype=torch.long).unsqueeze(0)
        image = torch.randn(3, 320, 480).unsqueeze(0)
        cnn_lstm = cnn_lstm.to(device)

        scores = cnn_lstm(image=image, text=text)
        self.assertEqual(scores.size(), torch.Size((1, 32)))
