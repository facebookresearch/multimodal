# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict, Generator, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.diffusion.predictors import Predictor

from torchmultimodal.modules.diffusion.schedules import DiffusionSchedule
from torchmultimodal.utils.diffusion_utils import DiffusionOutput


class DDPModule(nn.Module):
    """DDPModule acts as a wrapper module around an inner neural network. During training it uses the
    inner neural network to predict a single denoising step. When set to eval, calling forward will
    sample the entire diffusion schedule. This module follows the denoising diffusion process as
    described in "Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2006.11239) and "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672).

    Example:
        ddpm_model = DDPModule(model, schedule, predictor)
        prediction_t = ddpm_model(x_t, t)

        ddpm_model.eval()
        x_0 = ddpm_model(x_T, T)

    Code ref:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

    Attributes:
        model (nn.Module): prediction neural network
        schedule (DiffusionSchedule): defines diffusion of noise through time
        predictor (Predictor): predictor class to handle predictions depending on the model input
        eval_steps (Tensor): subset of steps to sample at inference

    Args:
        xt (Tensor): corrupted data at time t (when t = schedule.steps, xt is equivalent to noise)
        timestep (Tensor): diffusion step
        conditional_inputs (Dict): dictionary of context embeddings

    """

    def __init__(
        self,
        model: nn.Module,
        schedule: DiffusionSchedule,
        predictor: Predictor,
        eval_steps: Optional[Tensor] = None,
        progress_bar: bool = True,
    ):
        super().__init__()

        self.model = model
        self.train_schedule = schedule
        self.train_predictor = predictor
        self.progress_bar = progress_bar

        if eval_steps is None:
            eval_steps = torch.arange(self.train_schedule.steps)
            eval_steps_map = eval_steps
            self.eval_schedule = schedule
            self.eval_predictor = predictor
        else:
            # Special schedule for strided sampling from equation 19
            # in "Improved Denoising Diffusion Probabilistic Models"
            eval_steps, _ = eval_steps.sort()
            # eval_map maps from timestep in full schedule, to timestep in truncated eval scheule
            # e.g. if train has 1000 steps, and eval has 3, then t = 500 would map to eval timestep = 1
            eval_steps_map = torch.zeros(self.train_schedule.steps, dtype=torch.long)
            eval_steps_map[eval_steps] = torch.arange(len(eval_steps))

            # Compute cumulative product of only the alphas in the eval steps
            # Recompute betas based on these cumulative products and create a new schedule with these betas
            alphas_cumprod = schedule.alphas_cumprod[eval_steps]
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            beta_schedule = 1 - alphas_cumprod / alphas_cumprod_prev
            self.eval_schedule = copy.deepcopy(schedule)
            self.eval_schedule.betas = beta_schedule
            self.eval_predictor = copy.deepcopy(predictor)
            self.eval_predictor.schedule.betas = beta_schedule

        self.eval_steps: Tensor
        self.eval_steps_map: Tensor
        self.register_buffer("eval_steps", eval_steps.to(torch.long))
        self.register_buffer("eval_steps_map", eval_steps_map)

    def predict_parameters(
        self, input: DiffusionOutput, xt: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Given model predictions, corrupted data (x at step t) and noise (x at final step T),
        compute the predicted normal mean and log_variance for the given diffusion step t.

        Args:
            input (DiffusionOutput): model output values
            xt (Tensor): corrupted data at time t
            t (Tensor): int diffusion steps
        """
        pred, value = input.prediction, input.variance_value
        schedule = self.train_schedule if self.training else self.eval_schedule
        predictor = self.train_predictor if self.training else self.eval_predictor
        timestep = t if self.training else self.eval_steps_map[t]

        x0 = predictor.predict_x0(pred, xt, timestep)
        return schedule.q_posterior(x0, xt, timestep, value)

    def remove_noise(
        self, xt: Tensor, t: Tensor, c: Optional[Dict[str, Tensor]]
    ) -> Tensor:
        """Given corrupted data (x at step t) and noise (x at final step T), compute x denoised
        by one diffusion step. This is the function p(xt) from
        https://arxiv.org/abs/2006.11239.

        Args:
            xt (Tensor): corrupted data at time t
            t (Tensor): int diffusion steps
            c (Dict): dictionary of context embeddings
        """
        # Model outputs
        out = self.model(xt, t, c)
        mean, log_variance = self.predict_parameters(out, xt, t)

        # Predict x_{t-1}
        dtype = xt.dtype
        noise = self.train_schedule.sample_noise(xt)
        # Mask noise when t = 0; shape (b, 1, ..., 1) with same dims as xt
        nonzero_mask = (t != 0).to(dtype).view(-1, *([1] * (xt.dim() - 1)))
        # pyre-ignore
        return mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def generator(
        self, xt: Tensor, c: Optional[Dict[str, Tensor]] = None
    ) -> Generator[Tensor, None, None]:
        """Generate xt for each t in sample_steps"""
        for step in self.eval_steps.flip(0):
            t = step * torch.ones(xt.size(0), device=xt.device, dtype=torch.long)
            xt = self.remove_noise(xt, t, c)
            yield xt

    def forward(
        self,
        xt: Tensor,
        timestep: Optional[Tensor] = None,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> Union[DiffusionOutput, Tensor]:
        if self.training:
            if timestep is None:
                raise ValueError("Must provide a t value during training")
            out = self.model(xt, timestep, conditional_inputs)
            if not isinstance(out, DiffusionOutput):
                raise TypeError("Model is expected to output a DiffusionOutput class")
            if out.variance_value is not None:
                out.mean, out.log_variance = self.predict_parameters(out, xt, timestep)
            return out
        else:
            gen: Iterable = self.generator(xt, conditional_inputs)
            if self.progress_bar:
                # Lazy import so that we don't depend on tqdm.
                from tqdm.auto import tqdm

                gen = tqdm(gen, total=len(self.eval_steps))
            for x in gen:
                pass
            # pyre-ignore
            return x
