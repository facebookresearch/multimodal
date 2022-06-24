# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from test.test_utils import assert_expected

from torchmultimodal.utils.common import format_convnet_params, shift_dim, tensor_slice


def test_shift_dim():
    test_random_tensor = torch.randn(2, 2, 2, 2, 2)
    actual = shift_dim(test_random_tensor, 1, -1)
    expected = test_random_tensor.permute(0, 2, 3, 4, 1).contiguous()
    assert_expected(actual, expected)

    actual = shift_dim(test_random_tensor, -3, 3)
    expected = test_random_tensor.permute(0, 1, 3, 2, 4).contiguous()
    assert_expected(actual, expected)


class TestTensorSlice:
    @pytest.fixture(scope="class")
    def test_input(self):
        return torch.tensor([[[0, 1], [2, 3], [5, 6]]])

    def test_default(self, test_input):
        actual = tensor_slice(test_input, [0, 1, 0], [1, 1, 2])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

    def test_size_minus_one(self, test_input):
        """Test size -1"""
        actual = tensor_slice(test_input, [0, 1, 0], [1, -1, 2])
        expected = torch.tensor([[[2, 3], [5, 6]]])
        assert_expected(actual, expected)

    def test_uneven_begin_size(self, test_input):
        """Test uneven begin and size vectors"""
        actual = tensor_slice(test_input, [0, 1, 0], [1, 1])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

        actual = tensor_slice(test_input, [0, 1], [1, 1, 2])
        expected = torch.tensor([[[2, 3]]])
        assert_expected(actual, expected)

    @pytest.mark.xfail(raises=ValueError, reason="Invalid begin")
    def test_invalid_begin(self, test_input):
        tensor_slice(test_input, [-1, 1, 0], [1, 1, 2])

    @pytest.mark.xfail(raises=ValueError, reason="Invalid size")
    def test_invalid_size(self, test_input):
        tensor_slice(test_input, [0, 1, 0], [-2, 1, 2])


class TestFormatConvnetParams:
    @pytest.fixture(scope="class")
    def params(self):
        channels = (2, 2, 2)
        kernel = (2, 2, 2)
        strides = (1, 1, 1)
        n_dims = 3
        return channels, kernel, strides, n_dims

    def test_int_params(self, params):
        channels, kernel, strides, n_dims = params
        _, actual_kernel, actual_strides = format_convnet_params(
            channels, kernel[0], strides[0], n_dims
        )
        expected_kernel = kernel
        assert (
            actual_kernel == expected_kernel
        ), "kernel int -> tuple[tuple] is incorrect"
        expected_strides = strides
        assert (
            actual_strides == expected_strides
        ), "strides int -> tuple[tuple] is incorrect"

    def test_single_tuple_params(self, params):
        channels, kernel, strides, n_dims = params
        # Test length == dimensionality
        _, actual_kernel, actual_strides = format_convnet_params(
            channels, kernel, strides, n_dims
        )
        expected_kernel = (kernel,) * n_dims
        assert (
            actual_kernel == expected_kernel
        ), "kernel tuple -> tuple[tuple] is incorrect"
        expected_strides = (strides,) * n_dims
        assert (
            actual_strides == expected_strides
        ), "strides tuple -> tuple[tuple] is incorrect"
        # Test length == number of layers, should be untouched
        channels += (2,)
        kernel += (2,)
        strides += (1,)
        _, actual_kernel, actual_strides = format_convnet_params(
            channels, kernel, strides, n_dims
        )
        expected_kernel = kernel
        assert actual_kernel == expected_kernel, "kernel tuple -> tuple is incorrect"
        expected_strides = strides
        assert actual_strides == expected_strides, "strides tuple -> tuple is incorrect"

    def test_tuple_tuple_params(self, params):
        channels, kernel, strides, n_dims = params
        kernel = (kernel,) * n_dims
        strides = (strides,) * n_dims
        _, actual_kernel, actual_strides = format_convnet_params(
            channels, kernel, strides, n_dims
        )
        expected_kernel = kernel
        assert (
            actual_kernel == expected_kernel
        ), "kernel tuple[tuple] -> tuple[tuple] is incorrect"
        expected_strides = strides
        assert (
            actual_strides == expected_strides
        ), "strides tuple[tuple] -> tuple[tuple] is incorrect"

    def test_invalid_lengths(self, params):
        channels, kernel, strides, n_dims = params
        with pytest.raises(ValueError):
            _ = format_convnet_params(channels, kernel[:2], strides, n_dims)
        with pytest.raises(ValueError):
            _ = format_convnet_params(channels, kernel, strides[:2], n_dims)
        with pytest.raises(ValueError):
            _ = format_convnet_params(channels[:2], kernel, strides, n_dims - 1)
