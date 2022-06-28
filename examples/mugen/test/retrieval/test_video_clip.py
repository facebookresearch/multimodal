# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mugen.retrieval.video_clip import (
    Projection,
    TextEncoder,
    videoclip,
    VideoEncoder,
)

from test.test_utils import assert_expected, set_rng_seed


class TestTextEncoder:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        input_ids = torch.Tensor(
            [
                [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102],
                [101, 2117, 7820, 3793, 102, 0, 0, 0, 0],
            ]
        ).to(dtype=int)
        return input_ids

    def test_forward_pretrained(self, utils):
        input_ids = utils
        encoder = TextEncoder()
        out = encoder(input_ids)
        expected_sum = -13.6029
        assert_expected(actual=out.shape, expected=torch.Size([2, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )
        print(encoder.state_dict().keys())

    def test_forward_untrained(self, utils):
        input_ids = utils
        encoder = TextEncoder(pretrained=False)
        out = encoder(input_ids)
        expected_sum = 7.1526e-07
        assert_expected(actual=out.shape, expected=torch.Size([2, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )

    def test_attention_mask(self, utils):
        input_ids = utils
        encoder = TextEncoder()
        attention_mask = encoder.build_attention_mask(input_ids)
        assert_expected(
            actual=attention_mask,
            expected=torch.as_tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0]]
            ),
        )


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


class TestProjection:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        input = torch.randint(10, (2, 7)).float()
        proj = Projection(dim_in=7, dim_out=3)
        return proj, input

    def test_forward(self, utils):
        proj, input = utils
        out = proj(input)
        expected = torch.Tensor([[-1.2214, -0.0066, 1.2280], [-1.3886, 0.4626, 0.9260]])
        assert_expected(actual=out, expected=expected, rtol=0, atol=1e-4)


class TestVideoCLIPModel:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        input_text = torch.Tensor(
            [
                [101, 6315, 3793, 7099, 2005, 5604, 19204, 17629, 102],
                [101, 2117, 7820, 3793, 102, 0, 0, 0, 0],
            ]
        ).to(dtype=int)
        input_video = torch.randint(10, [2, 3, 32, 32, 32]).float()
        clip = videoclip()
        return clip, input_text, input_video

    def test_forward(self, utils):
        clip, input_text, input_video = utils
        clip_output = clip(features_a=input_text, features_b=input_video)
        assert_expected(
            actual=clip_output.embeddings_a.shape, expected=torch.Size([2, 256])
        )
        assert_expected(
            actual=clip_output.embeddings_b.shape, expected=torch.Size([2, 256])
        )
