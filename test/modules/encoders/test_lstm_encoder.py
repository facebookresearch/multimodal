# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.modules.encoders.lstm_encoder import LSTMEncoder


class TestLSTMEncoder(unittest.TestCase):
    def test_lstm_encoder(self):
        input = torch.randint(1, 79, (10,), dtype=torch.long).unsqueeze(0)
        lstm_encoder = LSTMEncoder(
            vocab_size=80,
            embedding_dim=20,
            input_size=20,
            hidden_size=50,
            bidirectional=True,
            batch_first=True,
        )
        out = lstm_encoder(input)
        self.assertEqual(out.size(), torch.Size([1, 100]))
