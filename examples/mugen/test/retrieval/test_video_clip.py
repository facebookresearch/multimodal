# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from examples.mugen.retrieval.video_clip import (
    build_videoclip,
    Projection,
    TextEncoder,
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

    def test_forward_pretrained_trainable(self, utils):
        input_ids = utils
        encoder = TextEncoder()
        out = encoder(input_ids)
        expected_sum = -13.6029
        assert_expected(actual=out.shape, expected=torch.Size([2, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )
        assert next(encoder.parameters()).requires_grad
        assert encoder.out_dim == 768

    def test_pretrained_untrainable(self):
        encoder = TextEncoder(trainable=False)
        assert not next(encoder.parameters()).requires_grad

    def test_forward_untrained_trainable(self, utils):
        input_ids = utils
        encoder = TextEncoder(pretrained=False)
        out = encoder(input_ids)
        expected_sum = 7.1526e-07
        assert_expected(actual=out.shape, expected=torch.Size([2, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )
        assert next(encoder.parameters()).requires_grad

    def test_untrained_untrainable(self):
        encoder = TextEncoder(pretrained=False, trainable=False)
        # encoder should ignore ``trainable`` if ``pretrained`` is False
        assert next(encoder.parameters()).requires_grad

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
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self):
        def make_input_video(c_dim=1):
            input_shape = [2, 3, 32, 32, 32]
            input_shape[c_dim] = 3
            return torch.randint(10, input_shape).float()

        return make_input_video

    def test_forward_trainable(self, utils):
        make_input_video = utils
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
        assert next(encoder.parameters()).requires_grad
        assert encoder.out_dim == 1024

    def test_untrainable(self):
        encoder = VideoEncoder(trainable=False)
        assert not next(encoder.parameters()).requires_grad


class TestProjection:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(1234)

    @pytest.fixture
    def utils(self, set_seed):
        input = torch.randint(10, (2, 7)).float()
        proj = Projection(in_dim=7, out_dim=3)
        return proj, input

    def test_forward(self, utils):
        proj, input = utils
        out = proj(input)
        expected = torch.Tensor([[-1.2214, -0.0066, 1.2280], [-1.3886, 0.4626, 0.9260]])
        assert_expected(actual=out, expected=expected, rtol=0, atol=1e-4)


class TestVideoCLIPBuilder:
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
        return input_text, input_video

    def test_forward(self, utils):
        input_text, input_video = utils
        model = build_videoclip()
        output = model(features_a=input_text, features_b=input_video)
        assert_expected(actual=output.embeddings_a.shape, expected=torch.Size([2, 256]))
        assert_expected(actual=output.embeddings_b.shape, expected=torch.Size([2, 256]))
