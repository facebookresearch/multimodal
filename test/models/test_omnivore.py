# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torchmultimodal.models.omnivore as omnivore

from test.test_utils import assert_expected, set_rng_seed
from torchmultimodal.utils.common import get_current_device


@pytest.fixture(autouse=True)
def device():
    set_rng_seed(42)
    return get_current_device()


@pytest.fixture(autouse=True)
def omnivore_swin_t_model(device):
    return omnivore.omnivore_swin_t().to(device)


@pytest.fixture(autouse=True)
def omnivore_swin_s_model(device):
    return omnivore.omnivore_swin_s().to(device)


@pytest.fixture(autouse=True)
def omnivore_swin_b_model(device):
    return omnivore.omnivore_swin_b().to(device)


def test_omnivore_swin_t_forward(omnivore_swin_t_model):
    model = omnivore_swin_t_model

    image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(torch.tensor(image_score.size()), torch.tensor((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(194.83563), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn(1, 4, 1, 112, 112)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(torch.tensor(rgbd_score.size()), torch.tensor((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(3.18015), rtol=1e-3, atol=1e-3)

    video = torch.randn(1, 3, 4, 112, 112)
    video_score = model(video, input_type="video")
    assert_expected(torch.tensor(video_score.size()), torch.tensor((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(100.87259), rtol=1e-3, atol=1e-3
    )


def test_omnivore_swin_s_forward(omnivore_swin_s_model):
    model = omnivore_swin_s_model

    image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(torch.tensor(image_score.size()), torch.tensor((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(240.41123), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn(1, 4, 1, 112, 112)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(torch.tensor(rgbd_score.size()), torch.tensor((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(5.73624), rtol=1e-3, atol=1e-3)

    video = torch.randn(1, 3, 4, 112, 112)
    video_score = model(video, input_type="video")
    assert_expected(torch.tensor(video_score.size()), torch.tensor((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(100.75939), rtol=1e-3, atol=1e-3
    )


def test_omnivore_swin_b_forward(omnivore_swin_b_model):
    model = omnivore_swin_b_model

    image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(torch.tensor(image_score.size()), torch.tensor((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(293.43484), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn(1, 4, 1, 112, 112)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(torch.tensor(rgbd_score.size()), torch.tensor((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(6.76342), rtol=1e-3, atol=1e-3)

    video = torch.randn(1, 3, 4, 112, 112)
    video_score = model(video, input_type="video")
    assert_expected(torch.tensor(video_score.size()), torch.tensor((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(131.65342), rtol=1e-3, atol=1e-3
    )


def test_omnivore_forward_wrong_input_type(omnivore_swin_t_model):
    model = omnivore_swin_t_model

    image = torch.randn(1, 3, 1, 112, 112)  # B C D H W
    with pytest.raises(AssertionError, match="Unsupported input_type: _WRONG_TYPE_.+"):
        _ = model(image, input_type="_WRONG_TYPE_")
