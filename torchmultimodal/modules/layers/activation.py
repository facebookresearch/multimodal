# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SiLU(nn.Module):
    r"""Sigmoid Linear Unit

    .. math:: \text{SiLU}(x) = x * \sigma(1.702 * x)

    where :math:`\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(1.702 * x) * x


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation function

    .. math:: \text{GEGLU}(a,b) = a * \text{GELU}(b)

    where :math:`a` is the first half of the input matrices and :math:`b` is
    the second half, as descibed in the paper:
    `"GLU Variants Improve Transformer"<https://arxiv.org/pdf/2002.05202.pdf>`.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.split_dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=self.split_dim)
        return x * F.gelu(gate)
