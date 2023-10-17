# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
)


class VLBLoss(nn.Module):
    """VLBLoss minimizes the KL divergence between the distribution of the forward diffusion process (noising) and the
    learned reverse process (denoising). Its name is derived from the Variational Lower Bound which is being optimized.
    This loss function can be used on it's own or used in conjunction with a simpler loss method as proposed by
    Nicol & Dhariwal 2021.

    The details of the loss function are described in "Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2006.11239) and "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672)

    Code ref:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time

    Args:
        pred_mean (Tensor): predicted mean for time t
        pred_log_variance (Tensor): predicted log variance for time t
        x0 (Tensor): target data point
        xt (Tensor): corrupted datapoint
        t (Tensor): diffusion step

    """

    def __init__(self, schedule: DiscreteGaussianSchedule):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.schedule = schedule

    def approx_standard_normal_cdf(self, x: Tensor) -> Tensor:
        return 0.5 * (
            1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x.pow(3))))
        )

    def discretized_gaussian_log_likelihood(
        self,
        x: Tensor,
        mean: Tensor,
        log_scale: Tensor,
        thres: float = 0.999,
        eps: float = 1e-12,
    ) -> Tensor:
        if not x.shape == mean.shape == log_scale.shape:
            ValueError("x, mean, and log_scale must all be the same shape")
        # TODO: lucidrain found eps = 1e-3 worked better for fp16
        centered_x = x - mean
        inv_stdv = torch.exp(-log_scale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = cdf_plus.clamp(min=eps).log()
        log_one_minus_cdf_min = (1.0 - cdf_min).clamp(min=eps).log()
        cdf_delta = cdf_plus - cdf_min

        log_probs = torch.where(
            x < -thres,
            log_cdf_plus,
            torch.where(
                x > thres, log_one_minus_cdf_min, cdf_delta.clamp(min=eps).log()
            ),
        )

        return log_probs

    def meanflat(self, x: Tensor) -> Tensor:
        return x.mean(dim=tuple(range(1, len(x.shape))))

    def normal_kl(
        self, x_mean: Tensor, x_log_var: Tensor, p_mean: Tensor, p_log_var: Tensor
    ) -> Tensor:
        return 0.5 * (
            -1.0
            + p_log_var
            - x_log_var
            + (x_log_var - p_log_var).exp()
            + ((x_mean - p_mean).pow(2)) * (-p_log_var).exp()
        )

    def forward(
        self,
        pred_mean: Tensor,
        pred_log_var: Tensor,
        x0: Tensor,
        xt: Tensor,
        t: Tensor,
    ) -> Tensor:
        # Compute Targets
        mean, log_variance = self.schedule.q_posterior(x0, xt, t)

        nat = 1.0 / math.log(2.0)
        kl = self.normal_kl(mean, log_variance, pred_mean, pred_log_var)
        kl = self.meanflat(kl) * nat

        decoder_nll = -self.discretized_gaussian_log_likelihood(
            x0, mean=pred_mean, log_scale=0.5 * pred_log_var
        )
        decoder_nll = self.meanflat(decoder_nll) * nat

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        losses = torch.where(t == 0, decoder_nll, kl)

        return losses.mean()
