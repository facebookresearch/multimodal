# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch import nn, Tensor


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        output = nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


def fp32layernorm(x: Tensor, layernorm: nn.Module) -> Tensor:
    """Supports mixed-precision training by casting to fp32 for layernorm and back"""
    if x.dtype != torch.float32:
        x_fp32 = x.float()
        x_fp32 = layernorm(x_fp32)
        return x_fp32.type_as(x)
    else:
        return layernorm(x)
