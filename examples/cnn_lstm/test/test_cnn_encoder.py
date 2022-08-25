# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from examples.cnn_lstm.cnn_encoder import CNNEncoder
from test.test_utils import assert_expected, set_rng_seed
from torch import nn, Tensor


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestCNNEncoder:
    @pytest.fixture()
    def input(self):
        return Tensor([1, 2, 3, 4, 5, 6, 1, 3, 5, 2, 4, 6]).reshape(2, 1, 2, 3)

    @pytest.fixture()
    def input_dims(self):
        return [0, 1, 2, 3]

    @pytest.fixture()
    def output_dims(self):
        return [1, 2, 4, 5]

    @pytest.fixture()
    def kernel_sizes(self):
        return [6, 7, 8, 9]

    @pytest.fixture()
    def single_layer_input(self):
        return torch.rand(3, 3, 2, 2)

    @pytest.fixture()
    def single_layer_cnn_encoder(self):
        return CNNEncoder([3], [3], [5])

    @pytest.fixture()
    def multiple_layer_input(self):
        return torch.rand(3, 3, 8, 8)

    @pytest.fixture()
    def multiple_layer_cnn_encoder(self):
        return CNNEncoder([3, 2, 1], [2, 1, 2], [3, 5, 7])

    @pytest.fixture()
    def small_cnn_encoder(self):
        return CNNEncoder([1], [1], [2])

    def test_invalid_arg_lengths(self, input_dims, output_dims, kernel_sizes):
        with pytest.raises(AssertionError):
            CNNEncoder(
                input_dims[1:],
                output_dims,
                kernel_sizes,
            )

    def test_invalid_output_dims(self, input_dims, output_dims, kernel_sizes):
        with pytest.raises(AssertionError):
            CNNEncoder(
                input_dims,
                output_dims,
                kernel_sizes,
            )

    def test_single_layer(self, single_layer_input, single_layer_cnn_encoder):
        actual = single_layer_cnn_encoder(single_layer_input)
        expected = Tensor(
            [
                [-0.452341, 0.680854, -0.557894],
                [-0.924794, 0.729902, -0.836271],
                [1.377135, -1.410758, 1.394166],
            ]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_multiple_layer(self, multiple_layer_input, multiple_layer_cnn_encoder):
        actual = multiple_layer_cnn_encoder(multiple_layer_input)
        expected = Tensor(
            [[-0.482730, -0.253406], [1.391524, 1.298026], [-0.908794, -1.044622]]
        )
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_fixed_weight_and_bias(self, input, small_cnn_encoder):
        small_cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.5]))
        small_cnn_encoder.cnn[0][0].weight = nn.Parameter(
            Tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0).unsqueeze(0)
        )
        actual = small_cnn_encoder(input)
        expected = Tensor([[-0.434959, 0.807781], [-1.429150, 1.056329]])
        assert_expected(actual, expected, rtol=0, atol=1e-4)

    def test_scripting(self, input, small_cnn_encoder):
        small_cnn_encoder.cnn[0][0].bias = nn.Parameter(Tensor([0.5]))
        small_cnn_encoder.cnn[0][0].weight = nn.Parameter(
            Tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0).unsqueeze(0)
        )
        scripted_encoder = torch.jit.script(small_cnn_encoder)
        actual = scripted_encoder(input)
        expected = Tensor([[-0.434959, 0.807781], [-1.429150, 1.056329]])
        assert_expected(actual, expected, rtol=0, atol=1e-4)
