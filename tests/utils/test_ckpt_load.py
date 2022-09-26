# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchmultimodal.utils.common import load_module_from_url


def test_load_module_from_url():
    model = nn.Linear(2, 3)
    load_module_from_url(
        model, "https://download.pytorch.org/models/multimodal/test/linear_2_3.pt"
    )
