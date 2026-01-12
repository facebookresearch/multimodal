# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tests.test_utils import assert_expected
from torchmultimodal.modules.layers.activation import GEGLU, SiLU


def test_sigmoid_linear_unit():
    silu = SiLU()
    actual = silu(torch.ones(3))
    expected = torch.tensor([0.8458, 0.8458, 0.8458])
    assert_expected(actual, expected)


def test_geglu():
    geglu = GEGLU()
    actual = geglu(torch.ones(10))
    expected = torch.tensor([0.8413, 0.8413, 0.8413, 0.8413, 0.8413])
    assert_expected(actual, expected, atol=1e-4, rtol=1e-5)
