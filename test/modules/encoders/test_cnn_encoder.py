# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn, Tensor
from test.test_utils import assert_expected
from torchmultimodal.modules.encoders.cnn_encoder import CNNEncoder
class TestCnnEncoder(unittest.TestCase):
    def test_invalid_arg_lengths(self):
        input_dims = [1,2]
        output_dims = [3,4,5]
        kernel_sizes = [6,7,8]
        self.assertRaises(
            AssertionError, CNNEncoder, input_dims, output_dims, kernel_sizes
        )

    def test_invalid_output_dims(self):
        input_dims = [0,1,2,3]
        output_dims = [1,2,4,5]
        kernel_sizes = [8,9,10,11]
        self.assertRaises(
            AssertionError, CNNEncoder, input_dims, output_dims, kernel_sizes
        )

    def test_single_layer_output_shape(self):
        input = torch.zeros(5,3,128,128)
        cnn_encoder = CNNEncoder([3],[3],[5])
        actual = cnn_encoder(input)
        expected = torch.zeros(5,12288)
        assert_expected(actual, expected)

    def test_multilayers_output_shape(self):
        input = torch.zeros(5,3,128,128)
        cnn_encoder = CNNEncoder([3,3,3],[3,3,3],[5,5,5])
        zero_bias = nn.Parameter(Tensor([0.,0.,0.]))
        for i in range(3):
            cnn_encoder.cnn[i][0].bias = zero_bias
        actual = cnn_encoder(input)
        expected = torch.zeros(5,768)
        assert_expected(actual, expected)

    def test_varying_dims_and_kernels(self):
        input = torch.rand(5,3,128,128)
        cnn_encoder = CNNEncoder([3,2,1],[2,1,2],[3,5,7])
        out = cnn_encoder(input)
        self.assertEqual(out.size(), torch.Size([5,512]))

    def test_fixed_weight_and_bias(self):
        input = Tensor(
            [
                [[[1,2,3],[4,5,6]]],
                [[[1,3,5],[2,4,6]]]
            ]
        )
        cnn_encoder = CNNEncoder([1],[1],[2])
        cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.]))
        cnn_encoder.cnn[0][0].weight = nn.Parameter(Tensor([[[[1.,1.],[1.,1.]]]]))
        actual = cnn_encoder(input)
        expected = Tensor(
            [
                [-0.6324555, 0.6324555],
                [-1.2649111, 1.2649111]
            ]
        )
        assert_expected(actual, expected)

    def test_scripting(self):
        input = Tensor(
            [
                [[[1,2,3],[4,5,6]]],
                [[[1,3,5],[2,4,6]]]
            ]
        )
        cnn_encoder = CNNEncoder([1],[1],[2])
        cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.]))
        cnn_encoder.cnn[0][0].weight = nn.Parameter(Tensor([[[[1.,1.],[1.,1.]]]]))
        scripted_encoder = torch.jit.script(cnn_encoder)
        actual = scripted_encoder(input)
        expected = Tensor(
            [
                [-0.6324555, 0.6324555],
                [-1.2649111, 1.2649111]
            ]
        )
        assert_expected(actual, expected)
