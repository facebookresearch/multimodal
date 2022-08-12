# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from test.test_utils import assert_expected, set_rng_seed
from torch import Tensor
from torchmultimodal.models.albef.text_encoder import ALBEFTextEncoder


@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(0)


@pytest.fixture(autouse=True)
def text_encoder():
    return ALBEFTextEncoder(hidden_size=3, num_attention_heads=1)


def test_text_encoder(text_encoder):
    input_ids = torch.randint(10, (2, 2))
    text_atts = Tensor([[1, 1], [1, 0]])
    output = text_encoder(input_ids, text_atts)
    expected = Tensor(
        [
            [[-0.824082, -0.583282, 1.407363], [-0.306520, 1.348891, -1.042372]],
            [[-0.925689, -0.463074, 1.388763], [-1.412740, 0.762259, 0.650481]],
        ]
    )
    assert_expected(output.last_hidden_state, expected, rtol=0, atol=1e-4)


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
