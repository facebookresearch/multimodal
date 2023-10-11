# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

from torch import nn, Tensor
from torchmultimodal.diffusion_labs.schedules.schedule import DiffusionSchedule


class RandomDiffusionSteps(nn.Module):
    """Data Transform to randomly sample noised data from the diffusion schedule.
    During diffusion training, random diffusion steps are sampled per model update.
    This transform samples steps and returns the steps (t), seed noise, and transformed
    data at time t (xt).

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        batched (bool): if True, transform expects a batched input

    Args:
        x (Tensor): data representing x0, artifact being learned. The 0 represents zero diffusion steps.
    """

    def __init__(self, schedule: DiffusionSchedule, batched: bool = True):
        super().__init__()
        self.schedule = schedule
        self.batched = batched

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.batched:
            t = self.schedule.sample_steps(x.unsqueeze(0))
            t = t.squeeze(0)
        else:
            t = self.schedule.sample_steps(x)
        noise = self.schedule.sample_noise(x)
        xt = self.schedule.q_sample(x, noise, t)
        return x, xt, noise, t
