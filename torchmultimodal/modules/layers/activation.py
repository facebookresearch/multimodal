# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
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
