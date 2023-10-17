# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from torch import nn
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
)


class ComputeV(nn.Module):
    """Data transform to compute v prediction target from x0 and noise. This transfrom
    is meant to be used with the VPredictor. V is first proposed
    in "Progressive Distillation for Fast Sampling of Diffusion Models" by Salimans
    and Ho (https://arxiv.org/abs/2202.00512).

    Attributes:
        schedule (DiscreteGaussianSchedule): defines diffusion of noise through time
        data_field (str): key name for the data to noise
        time_field (str): key name for the diffusion timestep
        noise_field (str): key name for the random noise
        v (str): key name for computed v tensor

    Args:
        x (Dict): data containing tensors "x", "t", and "noise".
    """

    def __init__(
        self,
        schedule: DiscreteGaussianSchedule,
        data_field: str = "x",
        time_field: str = "t",
        noise_field: str = "noise",
        v_field: str = "v",
    ):
        super().__init__()
        self.schedule = schedule
        self.x0 = data_field
        self.t = time_field
        self.noise = noise_field
        self.v = v_field

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        assert self.x0 in x, f"{type(self).__name__} expects key {self.x0}"
        assert self.t in x, f"{type(self).__name__} expects key {self.t}"
        assert self.noise in x, f"{type(self).__name__} expects key {self.noise}"
        x0, t, noise = x[self.x0], x[self.t], x[self.noise]
        shape, dtype = x0.shape, x0.dtype
        e_coef = self.schedule("sqrt_alphas_cumprod", t, shape)
        x_coef = self.schedule("sqrt_compliment_alphas_cumprod", t, shape)
        v = e_coef * noise - x_coef * x0
        x[self.v] = v.to(dtype)
        return x
