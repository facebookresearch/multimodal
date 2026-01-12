#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch
from tests.test_utils import assert_expected, set_rng_seed
from torch import nn
from torchmultimodal.diffusion_labs.modules.adapters.super_resolution import (
    SuperResolution,
)


@pytest.fixture(autouse=True)
def set_seed(seed: int = 1):
    set_rng_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def params():
    in_channels = 3
    s = 4
    embed_dim = 6
    return in_channels, s, embed_dim


class TestSuperRes:
    @pytest.fixture
    def cond_large(self, params):
        embed_dim = params[-1]
        c = torch.ones(1, 1, embed_dim, embed_dim)
        return {"low_res": c}

    @pytest.fixture
    def cond_small(self, params):
        embed_dim = params[-1]
        c = torch.ones(1, 1, embed_dim // 2, embed_dim // 2)
        return {"low_res": c}

    @pytest.fixture
    def timestep(self):
        return torch.ones(1)

    @pytest.fixture
    def model(self):
        class DummyUNet(nn.Module):
            def forward(self, x, t, c=None):
                return x + t

        return SuperResolution(DummyUNet())

    @pytest.mark.parametrize("cond", ["cond_large", "cond_small"])
    def test_superres_forward(self, model, params, timestep, cond, request):
        embed_dim = params[-1]
        x = torch.ones(1, 1, embed_dim, embed_dim)
        actual = model(x, timestep, request.getfixturevalue(cond))
        expected = 2 * torch.ones(1, 2, embed_dim, embed_dim)
        assert_expected(actual, expected)
