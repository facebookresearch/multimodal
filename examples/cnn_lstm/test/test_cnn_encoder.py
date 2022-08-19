# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from examples.cnn_lstm.cnn_encoder import CNNEncoder
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor


class TestCNNEncoder(unittest.TestCase):
    def setUp(self):
        set_rng_seed(0)
        self.input = Tensor([1, 2, 3, 4, 5, 6, 1, 3, 5, 2, 4, 6]).reshape(2, 1, 2, 3)
        self.input_dims = [0, 1, 2, 3]
        self.output_dims = [1, 2, 4, 5]
        self.kernel_sizes = [6, 7, 8, 9]

    def test_invalid_arg_lengths(self):
        self.assertRaises(
            AssertionError,
            CNNEncoder,
            self.input_dims[1:],
            self.output_dims,
            self.kernel_sizes,
        )

    def test_invalid_output_dims(self):
        self.assertRaises(
            AssertionError,
            CNNEncoder,
            self.input_dims,
            self.output_dims,
            self.kernel_sizes,
        )

    def test_single_layer(self):
        input = torch.rand(3, 3, 2, 2)
        cnn_encoder = CNNEncoder([3], [3], [5])
        actual = cnn_encoder(input)
        expected = Tensor(
            [
                [-0.452341, 0.680854, -0.557894],
                [-0.924794, 0.729902, -0.836271],
                [1.377135, -1.410758, 1.394166],
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_multiple_layer(self):
        input = torch.rand(3, 3, 8, 8)
        cnn_encoder = CNNEncoder([3, 2, 1], [2, 1, 2], [3, 5, 7])
        actual = cnn_encoder(input)
        expected = Tensor(
            [[-0.482730, -0.253406], [1.391524, 1.298026], [-0.908794, -1.044622]]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_fixed_weight_and_bias(self):
        cnn_encoder = CNNEncoder([1], [1], [2])
        cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.5]))
        cnn_encoder.cnn[0][0].weight = nn.Parameter(
            Tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0).unsqueeze(0)
        )
        actual = cnn_encoder(self.input)
        expected = Tensor([[-0.434959, 0.807781], [-1.429150, 1.056329]])
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scripting(self):
        cnn_encoder = CNNEncoder([1], [1], [2])
        cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.5]))
        cnn_encoder.cnn[0][0].weight = nn.Parameter(
            Tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0).unsqueeze(0)
        )
        scripted_encoder = torch.jit.script(cnn_encoder)
        actual = scripted_encoder(self.input)
        expected = Tensor([[-0.434959, 0.807781], [-1.429150, 1.056329]])
        assert_expected(actual, expected, rtol=0, atol=1e-4)
