#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
from torchmultimodal.diffusion_labs.utils.common import cascaded_resize


def test_cascaded_resize():
    image = Image.new("RGBA", size=(15, 20), color=(128, 0, 0))
    actual = sum(cascaded_resize(image, 3).size)
    expected = 8
    assert actual == expected, "Cascaded resize returning wrong size"
