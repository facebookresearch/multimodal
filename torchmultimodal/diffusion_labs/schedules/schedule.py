# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod, abstractproperty
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class DiffusionSchedule(Protocol):
    """Class that defines the entire diffusion process and provides helper functions
    for computing various transformations given the diffusion process
    """

    @abstractmethod
    def sample_noise(self, x_like: Tensor) -> Tensor:
        """Sample from diffusion distribution

        Args:
           x_like (Tensor): example tensor to get meta properties for noise tensor
        """

    @abstractmethod
    def sample_steps(self, x_like: Tensor) -> Tensor:
        """Sample diffusion steps

        Args:
           x_like (Tensor): example tensor to get meta properties for noise tensor
        """

    @abstractmethod
    def q_sample(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Given data (x at step 0) and noise, compute xt for the given
        diffusion t.

        Args:
            x0 (Tensor): uncorrupted data at step 0
            noise (Tensor): sample noise, same size as x0
            t (Tensor): int diffusion steps
        """

    @abstractproperty
    def steps(self) -> int:
        """Number of diffusion steps"""
