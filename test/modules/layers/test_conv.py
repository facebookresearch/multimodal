# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch
from test.test_utils import assert_expected
from torchmultimodal.modules.layers.conv import (
    calculate_same_padding,
    calculate_transpose_padding,
    SamePadConv3d,
    SamePadConvTranspose3d,
)


class TestSamePadConv3d(unittest.TestCase):
    """
    Test the SamePadConv3d class and associated helpers
    """

    def setUp(self):
        inputs = [torch.ones(1, 1, 8, 8, 8), torch.ones(1, 1, 7, 7, 7)]
        kernels = [(4, 4, 4), (3, 3, 3)]
        strides = [(2, 2, 2), (3, 3, 3)]
        self.test_cases = list(product(*[inputs, kernels, strides]))
        self.pad_expected = [
            (1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1),
            (1, 0, 1, 0, 1, 0),
            (1, 0, 1, 0, 1, 0),
            (2, 1, 2, 1, 2, 1),
            (2, 1, 2, 1, 2, 1),
            (1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1),
        ]
        self.out_shape_conv_expected = [
            torch.tensor([1, 1, 4, 4, 4]),
            torch.tensor([1, 1, 3, 3, 3]),
            torch.tensor([1, 1, 4, 4, 4]),
            torch.tensor([1, 1, 3, 3, 3]),
            torch.tensor([1, 1, 4, 4, 4]),
            torch.tensor([1, 1, 3, 3, 3]),
            torch.tensor([1, 1, 4, 4, 4]),
            torch.tensor([1, 1, 3, 3, 3]),
        ]
        self.out_shape_convtranspose_expected = [
            torch.tensor([1, 1, 16, 16, 16]),
            torch.tensor([1, 1, 24, 24, 24]),
            torch.tensor([1, 1, 16, 16, 16]),
            torch.tensor([1, 1, 24, 24, 24]),
            torch.tensor([1, 1, 14, 14, 14]),
            torch.tensor([1, 1, 21, 21, 21]),
            torch.tensor([1, 1, 14, 14, 14]),
            torch.tensor([1, 1, 21, 21, 21]),
        ]
        self.transpose_pad_expected = [
            (3, 3, 3),
            (4, 4, 4),
            (2, 2, 2),
            (2, 2, 2),
            (4, 4, 4),
            (5, 5, 5),
            (3, 3, 3),
            (3, 3, 3),
        ]
        self.output_pad_expected = [
            (0, 0, 0),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            (0, 0, 0),
        ]

    def test_calculate_same_padding_assert(self):
        with self.assertRaises(ValueError):
            _ = calculate_same_padding((3, 3), (2, 2, 2), (5, 5))
            _ = calculate_same_padding(3, (2, 2), (5, 5, 5))

    def test_calculate_same_padding_output(self):
        for i, (inp, kernel, stride) in enumerate(self.test_cases):
            pad_actual = calculate_same_padding(kernel, stride, inp.shape[2:])
            self.assertEqual(
                pad_actual,
                self.pad_expected[i],
                f"padding incorrect for shape {inp.shape}, kernel {kernel}, stride {stride}",
            )

    def test_samepadconv3d_forward(self):
        for i, (inp, kernel, stride) in enumerate(self.test_cases):
            conv = SamePadConv3d(1, 1, kernel, stride, padding=0)
            out = conv(inp)
            out_shape_conv_actual = torch.tensor(out.shape)
            assert_expected(out_shape_conv_actual, self.out_shape_conv_expected[i])

    def test_calculate_transpose_padding_assert(self):
        with self.assertRaises(ValueError):
            _ = calculate_transpose_padding((3, 3), (2, 2, 2), (5, 5))
            _ = calculate_transpose_padding(3, (2, 2), (5, 5, 5))
        with self.assertRaises(ValueError):
            _ = calculate_transpose_padding((3, 3), (2, 2), (5, 5), (1, 0, 1))
            _ = calculate_transpose_padding(3, 2, (5, 5, 5), (1, 1, 1, 1, 1, 1, 1))

    def test_calculate_transpose_padding_output(self):
        for i, (inp, kernel, stride) in enumerate(self.test_cases):
            pad = calculate_same_padding(kernel, stride, inp.shape[2:])
            transpose_pad_actual, output_pad_actual = calculate_transpose_padding(
                kernel, stride, inp.shape[2:], pad
            )
            self.assertEqual(
                transpose_pad_actual,
                self.transpose_pad_expected[i],
                f"transpose padding incorrect for shape {inp.shape}, kernel {kernel}, stride {stride}",
            )
            self.assertEqual(
                output_pad_actual,
                self.output_pad_expected[i],
                f"output padding incorrect for shape {inp.shape}, kernel {kernel}, stride {stride}",
            )

    def test_samepadconvtranspose3d_forward(self):
        for i, (inp, kernel, stride) in enumerate(self.test_cases):
            conv = SamePadConvTranspose3d(1, 1, kernel, stride)
            out = conv(inp)
            out_shape_convtranspose_actual = torch.tensor(out.shape)
            assert_expected(
                out_shape_convtranspose_actual, self.out_shape_convtranspose_expected[i]
            )
