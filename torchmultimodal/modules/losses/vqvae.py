# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torch import nn, Tensor
from torch.nn import functional as F


class CommitmentLoss(nn.Module):
    """Commitment loss calculates the mean Euclidean distance between pairs of encoder output vectors
    and their corresponding quantized vectors. It encourages an encoder to generate outputs closer to an embedding.
    This is the beta in Eq. 3 of Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)

    Args:
        commitment_cost (float): multiplicative weight for the commitment loss value
    """

    def __init__(self, commitment_cost: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, quantized: Tensor, encoded: Tensor) -> Tensor:
        # Quantized vectors must be detached because commitment loss only lets gradient flow through encoder output
        loss = F.mse_loss(quantized.detach(), encoded) * self.commitment_cost

        return loss
