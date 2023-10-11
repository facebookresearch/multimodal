# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Dict, Generator, Optional, Protocol, runtime_checkable, Union

from torch import nn, Tensor
from torchmultimodal.diffusion_labs.predictors.predictor import Predictor

from torchmultimodal.diffusion_labs.schedules.schedule import DiffusionSchedule
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


@runtime_checkable
class Sampler(Protocol):
    """Sampler class for applying the learned denoising function given the diffusion schedule.
    During training this class passes through the model outputs but during eval, the model loops
    the model for all the eval steps. This class implements the same forward signature as the
    Adapter class. To access individual generative steps at eval, the Sampler.generator() method
    will return a python generator that steps through each denoising step.

    Example:
        model = Sampler(...)
        x = torch.randn(...)

        # Generate with forward
        model.eval()
        with torch.no_grad():
            img = model(x)

        # Generator with generator
        model.eval()
        gen = model.generator(x)
        images = []
        with torch.no_grad():
            for i in gen:
                images.append(i)

        img == images[-1]
    """

    model: nn.Module
    schedule: DiffusionSchedule
    predictor: Predictor
    eval_steps: Tensor

    @abstractmethod
    def generator(
        self,
        x: Tensor,
        c: Optional[Dict[str, Tensor]] = None,
    ) -> Generator[Tensor, None, None]:
        """Generator for each t in self.eval_steps

        Args:
            x (Tensor): corrupted data at time t (when t = schedule.steps, x is fully noise)
                     of shape  [b, c, ...]
            c (Dict): dictionary of model conditional inputs
        """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        timestep: Optional[Tensor] = None,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> Union[DiffusionOutput, Tensor]:
        """nn Module forward method

        Args:
            x (Tensor): corrupted data at time t (when t = schedule.steps, x is fully noise)
                     of shape  [b, c, ...]
            timestep (Optional[Tensor]): diffusion step
            conditional_inputs (Dict): dictionary of model conditional inputs
        """
