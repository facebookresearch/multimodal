# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm


def test_fp32layernorm():
    x = torch.ones(1, 1, dtype=torch.float16)
    norm = Fp32LayerNorm(1)
    output = norm(x)
    assert output.dtype == torch.float16
