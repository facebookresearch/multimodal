# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from test.test_utils import assert_expected

from torchmultimodal.utils.common import shift_dim


class TestCommonUtils(unittest.TestCase):
    """
    Test the utils in common.py
    """

    def setUp(self):
        self.test_random_tensor = torch.randn(2, 2, 2, 2, 2)

    def test_shift_dim(self):
        actual = shift_dim(self.test_random_tensor, 1, -1)
        expected = self.test_random_tensor.permute(0, 2, 3, 4, 1).contiguous()
        assert_expected(actual, expected)

        actual = shift_dim(self.test_random_tensor, -3, 3)
        expected = self.test_random_tensor.permute(0, 1, 3, 2, 4).contiguous()
        assert_expected(actual, expected)
