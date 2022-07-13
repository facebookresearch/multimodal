# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_multimodal_encoder import (
    ALBEFMultimodalEncoder,
)


@pytest.fixture(autouse=True)
def multimodal_encoder():
    set_rng_seed(0)
    return ALBEFMultimodalEncoder(hidden_size=3, num_attention_heads=1)


def test_multimodal_encoder(multimodal_encoder):
    image_embeds = torch.randn(2, 2, 3)
    text_embeds = torch.randn(2, 2, 3)
    text_atts = torch.Tensor([[1, 1], [1, 0]])
    output = multimodal_encoder(image_embeds, text_embeds, text_atts)
    expected = Tensor(
        [
            [[0.794870, 0.615549, -1.410419], [1.314163, -1.109555, -0.204607]],
            [[-0.862034, -0.539896, 1.401930], [-1.176761, -0.090902, 1.267663]],
        ]
    )
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_invalid_image_hidden_size(multimodal_encoder):
    image_embeds = torch.randn(2, 2, 4)
    text_embeds = torch.randn(2, 2, 3)
    text_atts = torch.Tensor([[1, 1], [1, 0]])
    with pytest.raises(RuntimeError):
        multimodal_encoder(image_embeds, text_embeds, text_atts)


def test_invalid_text_hidden_size(multimodal_encoder):
    image_embeds = torch.randn(2, 2, 3)
    text_embeds = torch.randn(2, 2, 4)
    text_atts = torch.Tensor([[1, 1], [1, 0]])
    with pytest.raises(RuntimeError):
        multimodal_encoder(image_embeds, text_embeds, text_atts)


def test_not_matching_input_batch_size(multimodal_encoder):
    image_embeds = torch.randn(2, 2, 3)
    text_embeds = torch.randn(3, 2, 3)
    text_atts = torch.Tensor([[1, 1], [1, 0], [1, 1]])
    with pytest.raises(RuntimeError):
        multimodal_encoder(image_embeds, text_embeds, text_atts)
