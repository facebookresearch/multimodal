# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.modules.encoders.albef_text_encoder import ALBEFTextEncoder


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(0)


@pytest.fixture
def text_encoder():
    return ALBEFTextEncoder(hidden_size=3, num_attention_heads=1)


def test_text_encoder(text_encoder):
    input_ids = torch.randint(10, (2, 2))
    text_atts = Tensor([[1, 1], [1, 0]])
    output = text_encoder(input_ids, text_atts)
    expected = Tensor(
        [
            [[-0.846098, -0.558322, 1.404420], [-0.337862, 1.358211, -1.020349]],
            [[-0.911407, -0.480780, 1.392188], [-1.404428, 0.846044, 0.558384]],
        ]
    )
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_text_encoder_without_attention_mask(text_encoder):
    input_ids = torch.randint(10, (2, 2))
    output = text_encoder(input_ids)
    expected = Tensor(
        [
            [[-0.846098, -0.558322, 1.404420], [-0.337862, 1.358211, -1.020349]],
            [[-0.852249, -0.551247, 1.403495], [-0.715966, 1.414176, -0.698211]],
        ]
    )
    assert_expected(output, expected, rtol=0, atol=1e-4)


def test_invalid_input_length(text_encoder):
    input_ids = torch.randint(10, (2, 2, 3))
    text_atts = torch.randint(2, (2, 2, 3))
    with pytest.raises(RuntimeError):
        text_encoder(input_ids, text_atts)


def test_not_matching_attention_mask_shape(text_encoder):
    input_ids = torch.randint(10, (2, 2))
    text_atts = torch.randint(2, (2, 3))
    with pytest.raises(RuntimeError):
        text_encoder(input_ids, text_atts)
