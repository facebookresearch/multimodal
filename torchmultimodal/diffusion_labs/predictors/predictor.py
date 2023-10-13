# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Callable, Optional, Protocol, runtime_checkable

from torch import Tensor
from torchmultimodal.diffusion_labs.schedules.discrete_gaussian_schedule import (
    DiscreteGaussianSchedule,
)


@runtime_checkable
class Predictor(Protocol):
    """Helper class to help predict various parts of the diffusion process. Different
    implementations of each method are needed depending on what the model itself was
    trained to predict.
    """

    schedule: DiscreteGaussianSchedule
    clamp_func: Optional[Callable]

    @abstractmethod
    def predict_x0(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Predict x0

        Args:
            prediction (Tensor): model prediction
            xt (Tensor): noised data to step t
            t (Tensor): int diffusion step for xt
        """

    @abstractmethod
    def predict_noise(self, prediction: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Predict noise

        Args:
            prediction (Tensor): model prediction
            xt (Tensor): noised data to step t
            t (Tensor): int diffusion step for xt
        """
