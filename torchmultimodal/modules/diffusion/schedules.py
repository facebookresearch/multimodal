# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


class DiffusionSchedule:
    """Diffusion is a thermondynamic process of two substances intermingling, likewise diffusion probabilistic models represent
    the transformation of one distribution into another as a gradual stochastic process. Specifically, in Denoising Diffusion, we
    model the transformation from the Gaussian distribution to some data distribution over a number of time steps. DiffusionSchedule
    is a parameterized helper class to model the changing distribution from Gaussian (noise) to data (x). This is an implementation
    of a diffusion schedule with discrete time steps.

    DiffusionSchedule manages all timestep properties that are a function of the variance schedule (betas). For example, to
    calculate the compliment of betas (called alphas in the paper), you define alphas

    def alphas(schedule):
        return (1.0 - schedule.betas)

    and then call schedule.add_property("alphas", alphas). This will compute the values of alphas and add the attribute to schedule.
    You can safely use any other existing property to define your new property. If betas is updated (e.g. for a shorter inference
    time schedule) then all of these attributes will be automatically recomputed.

    The details of diffusion can be found in "Deep Unsupervised Learning using Nonequilibrium Thermodynamics
    (https://arxiv.org/abs/1503.03585) and "Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2006.11239)

    Code ref:
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils.py

    Attributes:
        betas (Tensor): Beta variance value for each time step
        variance_range_value (float): variance value for reverse diffusion process defined from 0.0 to 1.0. For each step the
                                      variance lower and upper bound is computed, and this value is used to set the interpolation
                                      between them.
    """

    def __init__(self, betas: Tensor, variance_range_value: float = 0.0):
        super().__init__()
        if not 0.0 <= variance_range_value <= 1.0:
            ValueError("Variance_range_value must be between 0 and 1")
        self._betas = betas
        self.variance_range_value = variance_range_value
        self.expressions: Dict[str, Callable] = {}

        self.add_property("alphas", _alphas)
        self.add_property("alphas_cumprod", _alphas_cumprod)
        self.add_property("alphas_cumprod_prev", _alphas_cumprod_prev)
        self.add_property("sqrt_alphas_cumprod", _sqrt_alphas_cumprod)
        self.add_property(
            "sqrt_compliment_alphas_cumprod", _sqrt_compliment_alphas_cumprod
        )

        # Lower and upper bound for forward process variances based on reverse process
        self.add_property("lower_posterior_log_variance", _lower_posterior_log_variance)
        self.add_property("upper_posterior_log_variance", _upper_posterior_log_variance)

        # Coefficients for posterior q function (EQ 11 from Improving DDPMs)
        self.add_property("posterior_mean_x0_coef", _posterior_mean_x0_coef)
        self.add_property("posterior_mean_xt_coef", _posterior_mean_xt_coef)

    def add_property(self, name: str, expression: Callable) -> None:
        """Define diffusion schedule properties as functions of schedule
        An expression should be of the form

        def <func_name>(schedule):
           return <expr>

        Lambdas aren't allowed as they cannot be pickled. The expression is
        stored so it can be computed as often as necessary. The property name
        is used to add an instance attribute the first time the user uses it.
        e.g. schedule.<name> returns <expression>(self)
        """
        assert name not in self.expressions, "Property already exists"
        assert (
            expression.__name__ != "<lambda>"
        ), "Properties must be named functions, lambdas are not picklable"
        # Expression added to dictionary, expression computed and lazily added as attribute <name> in __getattr__
        self.expressions[name] = expression

    def _clear_cache(self) -> None:
        """Clear cached properties, they will be lazily recomputed next time they're called"""
        for p in self.expressions.keys():
            # properties are cached as instance variables
            if p in vars(self):
                vars(self).pop(p)

    def sample_noise(self, x_like: Tensor) -> Tensor:
        """Sample from Gaussian distribution"""
        return torch.randn_like(x_like)

    def sample_steps(self, x_like: Tensor) -> Tensor:
        """Sample diffusion steps"""
        b = x_like.size(0)
        return torch.randint(0, self.steps, (b,), device=x_like.device, dtype=torch.int)

    def q_sample(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """Given data (x at step 0) and noise, compute xt for the given
        diffusion t. Function q(xt | x0) from
        https://arxiv.org/abs/2006.11239.

        Args:
            x0 (Tensor): uncorrupted data at step 0
            noise (Tensor): sample noise, same size as x0
            t (Tensor): int diffusion steps
        """
        shape, dtype = x0.shape, x0.dtype
        x_coef = self("sqrt_alphas_cumprod", t, shape)
        e_coef = self("sqrt_compliment_alphas_cumprod", t, shape)
        xt = x_coef * x0 + e_coef * noise
        return xt.to(dtype)

    def q_posterior(
        self,
        x0: Tensor,
        xt: Tensor,
        t: Tensor,
        variance_range_value: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Given data (x at step 0) and corrupted data (x at step t), compute normal
        mean and log_variance for the given diffusion step t. Computes mu(xt, x0) and
        Sigma(xt, t, v) where v is the interpolation value. Equation 13 and 15 from
        https://arxiv.org/abs/2102.09672.

        Args:
            x0 (Tensor): uncorrupted data at step 0
            xt (Tensor): noised data to step t
            t (Tensor): int diffusion step for xt
            variance_range_value (Tensor): variance range if provided by the model
        """
        if variance_range_value is None:
            variance_range_value = torch.ones_like(x0) * self.variance_range_value
        shape, dtype = xt.shape, xt.dtype
        min_log = self("lower_posterior_log_variance", t, shape)
        max_log = self("upper_posterior_log_variance", t, shape)
        log_variance = (
            variance_range_value * max_log + (1 - variance_range_value) * min_log
        )
        mean = (
            self("posterior_mean_x0_coef", t, shape) * x0
            + self("posterior_mean_xt_coef", t, shape) * xt
        )
        return mean.to(dtype), log_variance.to(dtype)

    @property
    def betas(self) -> Tensor:
        return self._betas

    @betas.setter
    def betas(self, betas: Tensor) -> None:
        self._betas = betas
        self._clear_cache()

    @property
    def steps(self) -> int:
        return len(self.betas)

    def __call__(
        self, var_name: str, t: Tensor, shape: Union[Tensor, torch.Size]
    ) -> Tensor:
        """Access a DiffusionSchedule property with a given shape at a given timestep"""
        tensor = getattr(self, var_name)
        # Ensure attribute is on same device as timestep tensor
        if tensor.device != t.device:
            tensor = tensor.to(t.device)
            setattr(self, var_name, tensor)
        b = t.numel()
        out = tensor.gather(-1, t.long())
        return out.reshape(b, *((1,) * (len(shape) - 1)))

    def __getattr__(self, name: str) -> Any:
        """Override object __getattr__ to lazily evaluate expressions.
        __getattr__ is called when __getattribute__ fails, i.e. when attribute <name> is not defined
        The first time a user accesses a property (schedule.<name>), __getattr__ will be called and
        compute the expression and save it as an instance attribute
        """
        if name in object.__getattribute__(self, "expressions"):
            value = object.__getattribute__(self, "expressions")[name](self)
            setattr(self, name, value)
            return value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


# **************************************** Scheduler Functions ****************************************
def _alphas(schedule: DiffusionSchedule) -> Tensor:
    return 1.0 - schedule.betas


def _alphas_cumprod(schedule: DiffusionSchedule) -> Tensor:
    return schedule.alphas.cumprod(axis=0)


def _alphas_cumprod_prev(schedule: DiffusionSchedule) -> Tensor:
    return F.pad(schedule.alphas_cumprod[:-1], (1, 0), value=1.0)


def _sqrt_alphas_cumprod(schedule: DiffusionSchedule) -> Tensor:
    return schedule.alphas_cumprod.sqrt()


def _sqrt_compliment_alphas_cumprod(schedule: DiffusionSchedule) -> Tensor:
    # pyre-ignore
    return (1.0 - schedule.alphas_cumprod).sqrt()


def _lower_posterior_log_variance(schedule: DiffusionSchedule) -> Tensor:
    # First element is 0 which has an infinite log (EQ 15 from Improving DDPMs)
    compliment_alphas_bar = 1.0 - schedule.alphas_cumprod
    compliment_alphas_bar_prev = 1.0 - schedule.alphas_cumprod_prev
    lpv = schedule.betas * compliment_alphas_bar_prev / compliment_alphas_bar
    # pyre-ignore
    lpv = torch.cat([lpv[1:2], lpv[1:]])
    return lpv.log()


def _upper_posterior_log_variance(schedule: DiffusionSchedule) -> Tensor:
    return schedule.betas.log()


def _posterior_mean_x0_coef(schedule: DiffusionSchedule) -> Tensor:
    alphas_cumprod_prev_sqrt = schedule.alphas_cumprod_prev.sqrt()
    compliment_alphas_cumprod = 1.0 - schedule.alphas_cumprod
    return schedule.betas * alphas_cumprod_prev_sqrt / compliment_alphas_cumprod


def _posterior_mean_xt_coef(schedule: DiffusionSchedule) -> Tensor:
    compliment_alphas_cumprod_prev = 1.0 - schedule.alphas_cumprod_prev
    alphas_sqrt = schedule.alphas.sqrt()
    compliment_alphas_cumprod = 1.0 - schedule.alphas_cumprod
    return compliment_alphas_cumprod_prev * alphas_sqrt / compliment_alphas_cumprod


# **************************************** Beta schedules ****************************************
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    Diffusion variances from step 0 to T, following a cosine function
    as proposed in "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5).pow(2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(
    timesteps: int, start: Optional[float] = None, end: Optional[float] = None
) -> Tensor:
    """
    Diffusion variances from step 0 to T, following a linear function
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001 if start is None else start
    beta_end = scale * 0.02 if end is None else end
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(
    timesteps: int, start: Optional[float] = None, end: Optional[float] = None
) -> Tensor:
    """
    Diffusion variances from step 0 to T, following a quadratic function
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001 if start is None else start
    beta_end = scale * 0.02 if end is None else end
    return torch.linspace(
        math.sqrt(beta_start), math.sqrt(beta_end), timesteps, dtype=torch.float64
    ).pow(2)


def sigmoid_beta_schedule(
    timesteps: int, start: Optional[float] = None, end: Optional[float] = None
) -> Tensor:
    """
    Diffusion variances from step 0 to T along a sigmoid curve
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001 if start is None else start
    beta_end = scale * 0.02 if end is None else end
    # the range [-6, 6] constitutes a majority of the sigmoid curve
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
