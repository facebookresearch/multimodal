# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected

from torchmultimodal.utils.common import shift_dim


def test_shift_dim():
    test_random_tensor = torch.randn(2, 2, 2, 2, 2)
    actual = shift_dim(test_random_tensor, 1, -1)
    expected = test_random_tensor.permute(0, 2, 3, 4, 1).contiguous()
    assert_expected(actual, expected)

    actual = shift_dim(test_random_tensor, -3, 3)
    expected = test_random_tensor.permute(0, 1, 3, 2, 4).contiguous()
    assert_expected(actual, expected)
