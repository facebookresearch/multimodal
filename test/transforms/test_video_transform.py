# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.transforms.video_transform import VideoTransform


class TestVideoTransform:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)
        input_videos = 255 * torch.rand(size=(20, 6, 4, 5, 3)).to(dtype=float)

        def assert_transforms(transformed_videos, time_samples):
            assert_expected(
                actual=transformed_videos.shape,
                expected=torch.Size([20, 3, time_samples, 224, 224]),
            )
            assert_expected(
                actual=torch.mean(transformed_videos),
                expected=torch.as_tensor(0.0),
                rtol=0,
                atol=5e-2,
            )
            assert_expected(
                actual=torch.std(transformed_videos),
                expected=torch.as_tensor(0.75),
                rtol=0,
                atol=5e-2,
            )

        return assert_transforms, input_videos

    def test_call(self, start):
        assert_transforms, input_videos = start
        transform = VideoTransform(
            time_samples=1,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.2857, 0.2857, 0.2857],
        )
        transformed_videos = transform(input_videos)
        assert_transforms(transformed_videos, 1)
