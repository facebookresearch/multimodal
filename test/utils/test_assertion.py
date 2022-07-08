# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchmultimodal.utils.assertion import assert_equal_lengths


class TestAssertEqualLengths:
    def test_different_lengths(self):
        with pytest.raises(ValueError):
            assert_equal_lengths([1], (1, 1))

    def test_same_lengths(self):
        assert_equal_lengths([1, 1], (1, 1))
