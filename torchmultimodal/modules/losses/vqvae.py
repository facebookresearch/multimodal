# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

from torch import nn, Tensor


@dataclass
class CommitmentLossOutput:
    loss: Tensor


class CommitmentLoss(nn.Module):
    def __init__(self, commitment_cost: float = 1.0, **kwargs: Any):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.commitment_cost = commitment_cost

    def forward(self, quantised: Tensor, encoded: Tensor):
        # Quantised vectors must be detached because commitment loss only lets gradient flow through encoder output
        loss = self.mse_loss(quantised.detach(), encoded) * self.commitment_cost

        return CommitmentLossOutput(loss=loss)
