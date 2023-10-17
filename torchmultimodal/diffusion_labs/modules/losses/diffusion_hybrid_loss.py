# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.modules.losses.vlb_loss import VLBLoss
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
)


class DiffusionHybridLoss(nn.Module):
    """
    Combines both simple loss (typically MSE) and VLB loss weighted by lambda, as described in Eq. 16 of
    "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
    VLB loss is only used to train the model learned variance.

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        simple_loss (nn.Module): loss function computed on prediction of diffusion model and desired target
            (typically noise). Default is nn.MSELoss.
        lmbda (float): lambda weight for vlb loss. Default is 0.001.

    Args:
        input (Tensor): prediction of diffusion model of shape [b, c, ...]
        target (Tensor): desired target of shape [b, c, ...]
        mean (Tensor): predicted mean of posterior/xt of shape [b, c, ...]
        log_variance (Tensor): predicted log variance of posterior/xt of shape [b, c, ...]
        x0 (Tensor): data sample of shape [b, c,...]
        xt (Tensor): noised data sample from diffusion process of shape [b, c, ...]
        t (Tensor): diffusion timesteps of shape [b, ]

    """

    def __init__(
        self,
        schedule: DiscreteGaussianSchedule,
        simple_loss: nn.Module = nn.MSELoss(),
        lmbda: float = 0.001,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.simple_loss = simple_loss
        self.vlb_loss = VLBLoss(schedule)
        self.lmbda = lmbda

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        mean: Tensor,
        log_variance: Tensor,
        x0: Tensor,
        xt: Tensor,
        t: Tensor,
    ) -> Tensor:
        # Detach mean as stop gradient for vlb loss
        # Weight the vlb loss smaller, for stability, when training in a hybrid setting using
        # another criterion to train the predictor as in the paper (recommended 0.001)
        return self.simple_loss(input, target) + self.lmbda * self.vlb_loss(
            mean.detach(), log_variance, x0, xt, t
        )
