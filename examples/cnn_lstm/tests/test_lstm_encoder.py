# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from examples.cnn_lstm.lstm_encoder import LSTMEncoder
from tests.test_utils import assert_expected


class TestLSTMEncoder:
    @pytest.fixture
    def input(self):
        return torch.randint(1, 79, (10,), dtype=torch.long).unsqueeze(0)

    @pytest.fixture
    def lstm_encoder(self):
        return LSTMEncoder(
            vocab_size=80,
            embedding_dim=20,
            input_size=20,
            hidden_size=50,
            bidirectional=True,
            batch_first=True,
        )

    def test_lstm_encoder(self, input, lstm_encoder):
        out = lstm_encoder(input)
        assert_expected(out.size(), torch.Size([1, 100]))
