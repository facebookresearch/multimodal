# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torch import Tensor
from torchmultimodal.diffusion_labs.predictors.predictor import Predictor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
)


class NoisePredictor(Predictor):
    """Given a model that's trained to predict diffusion noise and corresponding schedule,
        this class computes the predicted noise and x0 at step t.

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        clamp_func (Callable): function to clamp prediction values
    """

    def __init__(
        self, schedule: DiscreteGaussianSchedule, clamp_func: Optional[Callable] = None
    ):
        self.clamp_func = clamp_func
        schedule.add_property("sqrt_recip_alphas_cumprod", _sqrt_recip_alphas_cumprod)
        schedule.add_property(
            "sqrt_recipm1_alphas_cumprod", _sqrt_recipm1_alphas_cumprod
        )
        self.schedule = schedule

    def predict_x0(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        shape, dtype = xt.shape, xt.dtype
        x_coef = self.schedule("sqrt_recip_alphas_cumprod", t, shape)
        e_coef = self.schedule("sqrt_recipm1_alphas_cumprod", t, shape)
        x0 = x_coef * xt - e_coef * prediction
        if self.clamp_func is not None:
            x0 = self.clamp_func(x0)
        return x0.to(dtype)

    def predict_noise(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        return prediction


def _sqrt_recip_alphas_cumprod(schedule: DiscreteGaussianSchedule) -> Tensor:
    # pyre-ignore
    return (1.0 / schedule.alphas_cumprod).sqrt()


def _sqrt_recipm1_alphas_cumprod(schedule: DiscreteGaussianSchedule) -> Tensor:
    # pyre-ignore
    return (1.0 / schedule.alphas_cumprod - 1).sqrt()
