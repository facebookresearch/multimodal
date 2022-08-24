# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from examples.cnn_lstm.cnn_lstm import cnn_lstm_classifier
from test.test_utils import assert_expected, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(1234)


class TestCNNLSTMModule:
    @pytest.fixture
    def classifier_in_dim(self):
        return 450

    @pytest.fixture
    def num_classes(self):
        return 32

    @pytest.fixture
    def cnn_lstm(self, classifier_in_dim, num_classes):
        return cnn_lstm_classifier(
            text_vocab_size=80,
            text_embedding_dim=20,
            cnn_input_dims=[3, 64, 128, 128, 64, 64],
            cnn_output_dims=[64, 128, 128, 64, 64, 10],
            cnn_kernel_sizes=[7, 5, 5, 5, 5, 1],
            lstm_input_size=20,
            lstm_hidden_dim=50,
            lstm_bidirectional=True,
            lstm_batch_first=True,
            classifier_in_dim=classifier_in_dim,
            num_classes=num_classes,
        )

    @pytest.fixture
    def text(self):
        return torch.randint(1, 79, (10,), dtype=torch.long).unsqueeze(0)

    @pytest.fixture
    def image(self):
        return torch.randn(3, 320, 480).unsqueeze(0)

    def test_forward(self, text, image, cnn_lstm):
        assert isinstance(cnn_lstm, torch.nn.Module)
        scores = cnn_lstm({"image": image, "text": text})
        assert_expected(scores.size(), torch.Size((1, 32)))
