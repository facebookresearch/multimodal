# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sized


def assert_equal_lengths(
    *args: Sized, msg: str = "iterable arguments must have same length."
) -> None:
    lengths = set()
    for item in args:
        lengths.add(len(item))
    if len(lengths) != 1:
        raise ValueError(msg)
