# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torchmultimodal.models.omnivore as omnivore
from tests.test_utils import set_rng_seed
from torchmultimodal.utils.common import get_current_device


@pytest.fixture(autouse=True)
def device():
    set_rng_seed(42)
    return get_current_device()


@pytest.fixture()
def omnivore_swin_t_model(device):
    return omnivore.omnivore_swin_t().to(device)


def test_omnivore_forward_wrong_input_type(omnivore_swin_t_model, device):
    model = omnivore_swin_t_model

    image = torch.randn((1, 3, 1, 112, 112), device=device)  # B C D H W
    with pytest.raises(AssertionError, match="Unsupported input_type: _WRONG_TYPE_.+"):
        _ = model(image, input_type="_WRONG_TYPE_")
