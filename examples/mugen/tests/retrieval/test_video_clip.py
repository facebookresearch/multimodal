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
from tests.test_utils import assert_expected, get_asset_path, set_rng_seed
from torchmultimodal import _PATH_MANAGER
from torchmultimodal.utils.common import shift_dim


def patch_load_module_from_url(mocker):
    """Mock the ``load_module_from_url`` utility function used in ``videoclip()`` to allow
    loading truncated state dicts with ``strict=False``.
    """

    def patched_load_module_from_url(
        model: torch.nn.Module, url: str, strict: bool = True, progress: bool = True
    ) -> None:
        local_path = _PATH_MANAGER.get_local_path(url)
        if not torch.cuda.is_available():
            state_dict = torch.load(local_path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(local_path)
        model.load_state_dict(state_dict, strict=False)

    return mocker.patch(
        "examples.mugen.retrieval.video_clip.load_module_from_url",
        new=patched_load_module_from_url,
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
            input_shape = [1, 3, 16, 224, 224]
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
        expected_sum = 408.3521
        assert_expected(
            actual=out.shape, expected=torch.Size([1, 1024])
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
        input_video = torch.randint(10, [2, 3, 16, 224, 224]).float()
        return input_text, input_video

    def test_forward_pretrained_trainable(self, utils, mocker):
        input_text, input_video = utils
        patch_load_module_from_url(mocker)
        model = videoclip(
            video_pretrain_path=get_asset_path("S3D_sample.pt"), proj_out_dim=3
        )

        assert next(model.encoder_a.parameters()).requires_grad
        assert next(model.encoder_b.parameters()).requires_grad

        output = model(features_a=input_text, features_b=input_video)
        assert_expected(
            actual=output.embeddings_a,
            expected=torch.Tensor(
                [[-0.4496, -0.3655, 0.8150], [0.2190, -0.7907, 0.5717]]
            ),
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            actual=output.embeddings_b,
            expected=torch.Tensor(
                [[0.7291, -0.0462, -0.6829], [0.7157, -0.0175, -0.6982]],
            ),
            rtol=0,
            atol=1e-3,
        )

    def test_pretrained_untrainable(self, mocker):
        patch_load_module_from_url(mocker)
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
                [[0.8164, -0.4178, -0.3987], [0.8147, -0.4537, -0.3611]]
            ),
            rtol=0,
            atol=1e-3,
        )
        assert_expected(
            actual=output.embeddings_b,
            expected=torch.Tensor(
                [[-0.0199, 0.7168, -0.6970], [0.5802, 0.2075, -0.7876]]
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
