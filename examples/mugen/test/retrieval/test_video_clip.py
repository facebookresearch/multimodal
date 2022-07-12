# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import pytest
import torch
from examples.mugen.retrieval.video_clip import (
    Projection,
    TextEncoder,
    videoclip,
    VideoEncoder,
)

from test.test_utils import assert_expected, get_asset_path, set_rng_seed
from torchmultimodal.utils.common import shift_dim


def patch_load_model(mocker):
    """Mock the ``load_model`` function of ``VideoEncoder`` to allow loading truncated
    state dicts with ``strict=False``.
    """

    def patched_load_model(
        cls,
        pretrained_url: Optional[str],
        load_state_dict: bool = True,
        state_dict_key: Optional[str] = None,
    ):
        assert isinstance(
            cls, torch.nn.Module
        ), "load_model can only be called on an nn.Module instance"
        if os.path.exists(pretrained_url):
            state_dict = torch.load(pretrained_url)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_url, model_dir=cls.get_model_dir(pretrained_url)
            )
        if state_dict_key:
            state_dict = state_dict[state_dict_key]

        if load_state_dict:
            cls.load_state_dict(state_dict, strict=False)
        return state_dict

    return mocker.patch(
        "examples.mugen.retrieval.video_clip.VideoEncoder.load_model",
        new=patched_load_model,
    )


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

    def test_forward(self, utils):
        input_ids = utils
        encoder = TextEncoder()
        out = encoder(input_ids)
        expected_sum = 7.1526e-07
        assert_expected(actual=out.shape, expected=torch.Size([2, 768]), rtol=0, atol=0)
        assert_expected(
            actual=out.sum(), expected=torch.as_tensor(expected_sum), rtol=0, atol=1e-4
        )
        assert encoder.out_dim == 768

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
            input_video = torch.randint(10, input_shape).float()
            input_video = (
                shift_dim(input_video, 1, c_dim) if c_dim != 1 else input_video
            )
            return input_video

        return make_input_video

    def test_forward(self, utils):
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
        assert encoder.out_dim == 1024

    def test_invalid_channels(self, utils):
        make_input_video = utils
        input_video = make_input_video(c_dim=3)
        encoder = VideoEncoder()
        with pytest.raises(ValueError):
            encoder(input_video)


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

    def test_forward_pretrained_trainable(self, utils, mocker):
        input_text, input_video = utils
        patch_load_model(mocker)
        model = videoclip(
            video_pretrain_path=get_asset_path("S3D_sample.pt"), proj_out_dim=3
        )

        assert next(model.encoder_a.parameters()).requires_grad
        assert next(model.encoder_b.parameters()).requires_grad

        output = model(features_a=input_text, features_b=input_video)
        assert_expected(
            actual=output.embeddings_a,
            expected=torch.Tensor(
                [[-0.7332, 0.6777, 0.0556], [-0.7345, 0.6761, 0.0583]]
            ),
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            actual=output.embeddings_b,
            expected=torch.Tensor(
                [[0.7953, -0.5579, -0.2374], [0.8051, -0.2850, -0.5202]]
            ),
            rtol=0,
            atol=1e-3,
        )

    def test_pretrained_untrainable(self, mocker):
        patch_load_model(mocker)
        model = videoclip(
            text_trainable=False,
            video_trainable=False,
            video_pretrain_path=get_asset_path("S3D_sample.pt"),
            proj_out_dim=3,
        )

        assert not next(model.encoder_a.parameters()).requires_grad
        assert not next(model.encoder_b.parameters()).requires_grad

    def test_forward_untrained_trainable(self, utils):
        input_text, input_video = utils
        model = videoclip(text_pretrained=False, video_pretrained=False, proj_out_dim=3)

        assert next(model.encoder_a.parameters()).requires_grad
        assert next(model.encoder_b.parameters()).requires_grad

        output = model(features_a=input_text, features_b=input_video)
        assert_expected(
            actual=output.embeddings_a,
            expected=torch.Tensor(
                [[-0.3398, 0.8129, -0.4730], [-0.8151, 0.4487, 0.3664]]
            ),
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            actual=output.embeddings_b,
            expected=torch.Tensor(
                [[0.4003, -0.8164, 0.4162], [-0.2378, -0.5576, 0.7953]]
            ),
            rtol=0,
            atol=1e-3,
        )

    def test_untrained_untrainable(self):
        with pytest.warns(UserWarning):
            model = videoclip(
                text_pretrained=False,
                text_trainable=False,
                video_pretrained=False,
                video_trainable=False,
                proj_out_dim=3,
            )

        assert next(model.encoder_a.parameters()).requires_grad
        assert next(model.encoder_b.parameters()).requires_grad
