# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from test.test_utils import assert_expected
from torchmultimodal.modules.layers.activation import SiLU


def test_sigmoid_linear_unit():
    silu = SiLU()
    actual = silu(torch.ones(3))
    expected = torch.tensor([0.8458, 0.8458, 0.8458])
    assert_expected(actual, expected)
