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
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        input_videos = 255 * torch.rand(size=(2, 6, 4, 5, 3)).to(dtype=float)
        transform = VideoTransform(
            time_samples=1,
            mean=(0.5, 0.5, 0.5),
            std=(0.2857, 0.2857, 0.2857),
            resize_shape=(6, 7),
        )

        return transform, input_videos

    def test_call(self, utils):
        transform, input_videos = utils
        out = transform(input_videos)
        assert_expected(actual=out.shape, expected=torch.Size([2, 3, 1, 6, 7]))
        assert_expected(
            actual=out.mean(), expected=torch.as_tensor(0.0), rtol=0, atol=5e-2
        )

    def test_wrong_channels(self, utils):
        transform, input_videos = utils
        with pytest.raises(ValueError):
            transform(input_videos[:, :, :, :, :2])  # only two channels

    def test_sample_frames(self, utils):
        transform, input_videos = utils
        out = transform.sample_frames(input_videos)
        assert_expected(actual=out.shape, expected=torch.Size([2, 1, 4, 5, 3]))

    def test_resize_hw(self, utils):
        transform, input_videos = utils
        out = transform.resize_hw(input_videos)
        assert_expected(actual=out.shape, expected=torch.Size([2, 6, 6, 7, 3]))

    def test_normalize(self, utils):
        transform, input_videos = utils
        out = transform.normalize(input_videos)
        assert_expected(actual=out.shape, expected=torch.Size([2, 6, 4, 5, 3]))
        assert_expected(
            actual=out.mean(), expected=torch.as_tensor(0.0), rtol=0, atol=5e-2
        )
        assert_expected(
            actual=out.std(), expected=torch.as_tensor(1.0), rtol=0, atol=5e-2
        )
