# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mugen.retrieval.video_clip import VideoEncoder

from test.test_utils import assert_expected, set_rng_seed


class TestVideoEncoder:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)

        def make_input_video(c_dim=1):
            input_shape = [2, 3, 32, 32, 32]
            input_shape[c_dim] = 3
            return torch.randint(10, input_shape).float()

        return make_input_video

    def test_forward(self, start):
        make_input_video = start
        input_video = make_input_video()
        encoder = VideoEncoder()
        out = encoder(input_video)
        expected_sum = 846.3781
        assert_expected(
            actual=out.shape, expected=torch.Size([2, 1024])
        )  # batch x embedding
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-3
        )
