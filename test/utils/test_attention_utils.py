# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from test.test_utils import assert_expected
from torchmultimodal.utils.attention import get_causal_attention_mask


def test_get_causal_attention_masks():
    actual = get_causal_attention_mask(3, 2)
    expected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    assert_expected(actual, expected)

    actual = get_causal_attention_mask(3, 3)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert_expected(actual, expected)
