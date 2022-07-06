# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mugen.retrieval.s3d import BasicConv3d, MixedConvsBlock, S3D, SepConv3d
from test.test_utils import assert_expected, set_rng_seed


class TestS3D:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        def make_input_video_from_shape(shape):
            return torch.randint(10, shape).float()

        def make_input_video():
            input_shape = [2, 3, 1, 5, 3]
            return make_input_video_from_shape(input_shape)

        return make_input_video, make_input_video_from_shape

    def test_forward(self, utils):
        _, make_input_video_from_shape = utils
        input_video = make_input_video_from_shape((2, 3, 32, 32, 32))
        s3d = S3D(num_class=3)
        out = s3d(input_video)
        assert_expected(
            actual=out.shape, expected=torch.Size([2, 3])
        )  # batch x num_class

    def test_basicconv3d(self, utils):
        make_input_video, _ = utils
        input_video = make_input_video()
        bc3 = BasicConv3d(3, 3, kernel_size=1, stride=1).float()
        out = bc3(input_video)
        assert_expected(actual=out.shape, expected=torch.Size([2, 3, 1, 5, 3]))

    def test_sepconv3d(self, utils):
        make_input_video, _ = utils
        input_video = make_input_video()
        sc3 = SepConv3d(3, 8, kernel_size=3, stride=2, padding=1)
        out = sc3(input_video)
        assert_expected(actual=out.shape, expected=torch.Size([2, 8, 1, 3, 2]))

    def test_mixedconvsblock(self, utils):
        make_input_video, _ = utils
        input_video = make_input_video()
        model = MixedConvsBlock(3, 1, 2, 2, 3, 3, 4)
        out = model(input_video)
        assert_expected(actual=out.shape, expected=torch.Size([2, 10, 1, 5, 3]))
