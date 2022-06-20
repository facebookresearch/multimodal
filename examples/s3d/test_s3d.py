# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest
import torch
from examples.s3d.s3d import BasicConv3d, S3D, S3DEncoder, SepConv3d
from test.test_utils import assert_expected, set_rng_seed


class TestS3D:
    @pytest.fixture
    def start(self):
        set_rng_seed(1234)

        def make_input_video_from_shape(shape):
            return torch.randint(10, shape).float()

        def make_input_video(num_channels=3, c_dim=1):
            input_shape = [2, 4, 1, 5, 3]
            input_shape[c_dim] = num_channels
            return make_input_video_from_shape(input_shape)

        return make_input_video, make_input_video_from_shape

    def test_s3d_encoder(self, start):
        make_input_video, _ = start
        input_video = make_input_video(c_dim=-1)
        s3d_e = S3DEncoder(trainable=True)
        out = s3d_e(input_video)
        expected_sum = 809.0687
        assert_expected(
            actual=out.shape, expected=torch.Size([2, 1024])
        )  # batch x embedding
        if abs(out.sum() - expected_sum) > 5e-4:
            raise AssertionError(f"Expected: {expected_sum}, Actual: {out.sum()}")

    def test_s3d_base(self, start):
        _, make_input_video_from_shape = start
        input_video = make_input_video_from_shape((2, 3, 32, 32, 32))
        s3d = S3D(num_class=3)
        out = s3d(input_video)
        assert_expected(
            actual=out.shape, expected=torch.Size([2, 3])
        )  # batch x num_class

    def test_basicconv3d(self, start):
        make_input_video, _ = start
        input_video = make_input_video(num_channels=4)
        bc3 = BasicConv3d(4, 4, kernel_size=1, stride=1).float()
        out = bc3(input_video)
        assert_expected(actual=out.shape, expected=torch.Size([2, 4, 1, 5, 3]))

    def test_sepconv3d(self, start):
        make_input_video, _ = start
        input_video = make_input_video(num_channels=4)
        sc3 = SepConv3d(4, 8, kernel_size=3, stride=2, padding=1)
        out = sc3(input_video)
        assert_expected(actual=out.shape, expected=torch.Size([2, 8, 1, 3, 2]))

    @pytest.mark.parametrize(
        "model_name,in_channels,out_channels",
        [
            ("Mixed3b", 192, 256),
            ("Mixed3c", 256, 480),
            ("Mixed4b", 480, 512),
            ("Mixed4c", 512, 512),
            ("Mixed4d", 512, 512),
            ("Mixed4e", 512, 528),
            ("Mixed4f", 528, 832),
            ("Mixed5b", 832, 832),
            ("Mixed5c", 832, 1024),
        ],
    )
    def test_mixed(self, start, model_name, in_channels, out_channels):
        make_input_video, _ = start
        input_video = make_input_video(num_channels=in_channels)
        module = importlib.import_module("examples.s3d.s3d")
        class_ = getattr(module, model_name)
        model = class_().float()
        out = model(input_video)
        assert_expected(
            actual=out.shape, expected=torch.Size([2, out_channels, 1, 5, 3])
        )
