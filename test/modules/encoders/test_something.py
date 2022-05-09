# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class TestSomething(unittest.TestCase):
    def setUp(self):
        pass

    def test_something(self):
        assert 1 == 2, "Something is broken here."
