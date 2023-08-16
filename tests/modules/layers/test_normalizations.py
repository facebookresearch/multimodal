# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchmultimodal.modules.layers.normalizations import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    RMSNorm,
)

from tests.test_utils import gpu_test


def test_fp32layernorm():
    x = torch.ones(1, 1, dtype=torch.float16)
    norm = Fp32LayerNorm(1)
    output = norm(x)
    assert output.dtype == torch.float16


def test_fp32groupnorm():
    x = torch.ones(2, 4, dtype=torch.float16)
    norm = Fp32GroupNorm(2, 4)
    output = norm(x)
    assert output.dtype == torch.float16


def test_RMSNorm_fp32return():
    """verify type is returned as fp32"""
    dims = 512
    x = torch.empty(dims, dtype=torch.float16)
    norm = RMSNorm(
        dims,
    )
    output = norm(x)
    assert output.dtype == torch.float32


@gpu_test(1)
def test_RMSNorm_core_algo():
    """compare RMSNorm with RMSNorm using F.norm version"""

    dims = 1024
    x = torch.empty(dims, dtype=torch.float16, device="cuda")
    x2 = x.clone().detach()

    class RMSNormFunctional(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = dim**0.5
            self.weights = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return F.normalize(x, dim=-1, eps=self.eps) * self.scale * self.weights

    baseNorm = RMSNorm(
        dims,
    ).to("cuda")
    backupNorm = RMSNormFunctional(
        dims,
    ).to("cuda")

    output_base = baseNorm(x)
    output_backup = backupNorm(x2)

    assert torch.allclose(output_base, output_backup)
