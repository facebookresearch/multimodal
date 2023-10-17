#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchmultimodal.diffusion_labs.transforms.super_resolution_transform import (
    SuperResolutionTransform,
)


def test_superres_transform():
    data_field = "x"
    small_name = "low_res"
    small_size = 16
    large_size = 32
    channels = 3
    batch_size = 2

    superres_transform = SuperResolutionTransform(
        low_res_size=small_size, size=large_size
    )
    sample = {data_field: torch.Tensor(batch_size, channels, large_size, large_size)}
    output = superres_transform(sample)

    small_img = output[small_name]
    assert small_img.shape == (batch_size, channels, large_size, large_size)
