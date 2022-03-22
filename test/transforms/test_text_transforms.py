# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchmultimodal.transforms.text_transforms import PadTransform, StrToIntTransform


class TestTextTransforms(unittest.TestCase):
    def setUp(self):
        self.input_1d_tensor = torch.ones(5)
        self.input_2d_tensor = torch.ones((8, 5))
        self.input_1d_string_list = ["1", "2", "3", "4", "5"]
        self.input_2d_string_list = [["1", "2", "3"], ["4", "5", "6"]]

    def test_pad_transform_long(self):
        pad_long = PadTransform(max_length=7)

        padded_1d_tensor_actual = pad_long(self.input_1d_tensor)
        padded_1d_tensor_expected = torch.cat([torch.ones(5), torch.zeros(2)])
        torch.testing.assert_close(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
            msg=f"actual: {padded_1d_tensor_actual}, expected: {padded_1d_tensor_expected}",
        )

        padded_2d_tensor_actual = pad_long(self.input_2d_tensor)
        padded_2d_tensor_expected = torch.cat(
            [torch.ones(8, 5), torch.zeros(8, 2)], axis=-1
        )
        torch.testing.assert_close(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
            msg=f"actual: {padded_2d_tensor_actual}, expected: {padded_2d_tensor_expected}",
        )

    def test_pad_transform_short(self):
        pad_short = PadTransform(max_length=3)

        padded_1d_tensor_actual = pad_short(self.input_1d_tensor)
        padded_1d_tensor_expected = self.input_1d_tensor
        torch.testing.assert_close(
            padded_1d_tensor_actual,
            padded_1d_tensor_expected,
            msg=f"actual: {padded_1d_tensor_actual}, expected: {padded_1d_tensor_expected}",
        )

        padded_2d_tensor_actual = pad_short(self.input_2d_tensor)
        padded_2d_tensor_expected = self.input_2d_tensor
        torch.testing.assert_close(
            padded_2d_tensor_actual,
            padded_2d_tensor_expected,
            msg=f"actual: {padded_2d_tensor_actual}, expected: {padded_2d_tensor_expected}",
        )

    def test_clip_multi_transform(self):
        str_to_int = StrToIntTransform()

        expected_1d_int_list = [1, 2, 3, 4, 5]
        actual_1d_int_list = str_to_int(self.input_1d_string_list)
        self.assertListEqual(expected_1d_int_list, actual_1d_int_list)

        expected_2d_int_list = [[1, 2, 3], [4, 5, 6]]
        actual_2d_int_list = str_to_int(self.input_2d_string_list)
        for i in range(len(expected_2d_int_list)):
            self.assertListEqual(expected_2d_int_list[i], actual_2d_int_list[i])
