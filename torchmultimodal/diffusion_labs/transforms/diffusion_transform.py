# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from torch import nn
from torchmultimodal.diffusion_labs.schedules.schedule import DiffusionSchedule


class RandomDiffusionSteps(nn.Module):
    """Data Transform to randomly sample noised data from the diffusion schedule.
    During diffusion training, random diffusion steps are sampled per model update.
    This transform samples steps and returns the steps (t), seed noise (noise), and transformed
    data at time t (xt).

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        batched (bool): if True, transform expects a batched input
        data_field (str): key name for the data to noise
        time_field (str): key name for the diffusion timestep
        noise_field (str): key name for the random noise
        noised_data_field (str): key name for the noised data

    Args:
        x (Dict): data containing tensor "x". This represents x0, the artifact being learned.
                  The 0 represents zero diffusion steps.
    """

    def __init__(
        self,
        schedule: DiffusionSchedule,
        batched: bool = True,
        data_field: str = "x",
        time_field: str = "t",
        noise_field: str = "noise",
        noised_data_field: str = "xt",
    ):
        super().__init__()
        self.schedule = schedule
        self.batched = batched
        self.x0 = data_field
        self.t = time_field
        self.noise = noise_field
        self.xt = noised_data_field

    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        assert self.x0 in x, f"{type(self).__name__} expects key {self.x0}"
        x0 = x[self.x0]
        if not self.batched:
            t = self.schedule.sample_steps(x0.unsqueeze(0))
            t = t.squeeze(0)
        else:
            t = self.schedule.sample_steps(x0)
        noise = self.schedule.sample_noise(x0)
        xt = self.schedule.q_sample(x0, noise, t)
        x.update(
            {
                self.t: t,
                self.noise: noise,
                self.xt: xt,
            }
        )
        return x
