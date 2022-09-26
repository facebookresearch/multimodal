# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torchmultimodal.models.omnivore as omnivore

from tests.test_utils import assert_expected, set_rng_seed
from torchmultimodal.utils.common import get_current_device


@pytest.fixture(autouse=True)
def device():
    set_rng_seed(42)
    return get_current_device()


@pytest.fixture()
def omnivore_swin_t_model(device):
    return omnivore.omnivore_swin_t().to(device)


@pytest.fixture()
def omnivore_swin_s_model(device):
    return omnivore.omnivore_swin_s().to(device)


@pytest.fixture()
def omnivore_swin_b_model(device):
    return omnivore.omnivore_swin_b().to(device)


def test_omnivore_swin_t_forward(omnivore_swin_t_model, device):
    model = omnivore_swin_t_model

    image = torch.randn((1, 3, 1, 112, 112), device=device)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(image_score.size(), torch.Size((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(200.27574), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn((1, 4, 1, 112, 112), device=device)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(rgbd_score.size(), torch.Size((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(3.10466), rtol=1e-3, atol=1e-3)

    video = torch.randn((1, 3, 4, 112, 112), device=device)
    video_score = model(video, input_type="video")
    assert_expected(video_score.size(), torch.Size((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(97.57287), rtol=1e-3, atol=1e-3
    )


def test_omnivore_swin_s_forward(omnivore_swin_s_model, device):
    model = omnivore_swin_s_model

    image = torch.randn((1, 3, 1, 112, 112), device=device)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(image_score.size(), torch.Size((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(238.41870), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn((1, 4, 1, 112, 112), device=device)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(rgbd_score.size(), torch.Size((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(4.05502), rtol=1e-3, atol=1e-3)

    video = torch.randn((1, 3, 4, 112, 112), device=device)
    video_score = model(video, input_type="video")
    assert_expected(video_score.size(), torch.Size((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(114.26146), rtol=1e-3, atol=1e-3
    )


def test_omnivore_swin_b_forward(omnivore_swin_b_model, device):
    model = omnivore_swin_b_model

    image = torch.randn((1, 3, 1, 112, 112), device=device)  # B C D H W
    image_score = model(image, input_type="image")

    assert_expected(image_score.size(), torch.Size((1, 1000)))
    assert_expected(
        image_score.abs().sum(), torch.tensor(234.38349), rtol=1e-3, atol=1e-3
    )

    rgbd = torch.randn((1, 4, 1, 112, 112), device=device)
    rgbd_score = model(rgbd, input_type="rgbd")
    assert_expected(rgbd_score.size(), torch.Size((1, 19)))
    assert_expected(rgbd_score.abs().sum(), torch.tensor(5.37857), rtol=1e-3, atol=1e-3)

    video = torch.randn((1, 3, 4, 112, 112), device=device)
    video_score = model(video, input_type="video")
    assert_expected(video_score.size(), torch.Size((1, 400)))
    assert_expected(
        video_score.abs().sum(), torch.tensor(128.43623), rtol=1e-3, atol=1e-3
    )


def test_omnivore_forward_wrong_input_type(omnivore_swin_t_model, device):
    model = omnivore_swin_t_model

    image = torch.randn((1, 3, 1, 112, 112), device=device)  # B C D H W
    with pytest.raises(AssertionError, match="Unsupported input_type: _WRONG_TYPE_.+"):
        _ = model(image, input_type="_WRONG_TYPE_")
