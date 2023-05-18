# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Generator, Iterable, Optional, Union

import torch
from torch import nn, Tensor
from torchmultimodal.modules.diffusion.predictors import Predictor

from torchmultimodal.modules.diffusion.schedules import DiffusionSchedule
from torchmultimodal.utils.diffusion_utils import DiffusionOutput


class DDIModule(nn.Module):
    """
    DDIModule implements "Denoising Diffusion Implicit Models" presented by Song et. al
    (https://arxiv.org/abs/2010.02502).

    DDIMs are a class of iterative implicit probabilistic models that has the same
    training procedure as DDPMs, but can produce high quality samples much faster.
    Song et. al propose a new generative process where given a noisy x_t, we first
    make a prediction for x_0, and we then use it to obtain a sample for x_{t−1}.
    DDIMs are known to be effective in generating high quality samples faster than
    DDPMs.

    Example:
        ddim_model = DDPModule(model, schedule, predictor)
        prediction_t = ddim_model(x_t, t)

        ddim_model.eval()
        x_0 = ddim_model(x_T, T)

    Code ref:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

    Attributes:
    model (nn.Module):
    schedule (DiffusionSchedule): defines noise diffusion throughout time
    predictor (Predictor): used to help predict x0
    eval_steps (Tensor): a subset of steps to sample at inference time
    eta (float): scaling factor used in Equation 12 of Song et. al
                    (https://arxiv.org/abs/2010.02502)
    Args:
        xt (Tensor): corrupted data at time t (when t = schedule.steps, xt is fully noise)
                     of shape  [b, c, ...]
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
        eta: float = 1.0,
    ):
        super().__init__()

        self.model = model
        self.schedule = schedule
        self.predictor = predictor
        self.progress_bar = progress_bar
        self.eta = eta

        if eval_steps is None:
            eval_steps = torch.arange(self.schedule.steps)
        else:
            eval_steps, _ = eval_steps.sort()

        self.eval_steps: Tensor
        self.register_buffer("eval_steps", eval_steps.to(torch.long))

    def remove_noise(
        self,
        xt: Tensor,
        c: Optional[Dict[str, Tensor]],
        cur_step: Tensor,
        next_step: Tensor,
    ) -> Tensor:
        """Given corrupted data (x at step t), compute x, denoised by one diffusion step,
        x_{t-1}.


        Args:
            xt (Tensor): uncorrupted data at step 0
            t (Tensor): diffusion step
            c (Optional[Dict[str, Tensor]]):
            cur_step Tensor:
            next_step Tensor:
            eta float:
        """
        # Model outputs
        alpha_bar = self.schedule("alphas_cumprod", cur_step, xt.shape)
        alpha_bar_next = self.schedule("alphas_cumprod", next_step, xt.shape)
        alpha_bar_next_sqred = self.schedule("sqrt_alphas_cumprod", next_step, xt.shape)

        out = self.model(xt, cur_step, c)
        pred = out.prediction

        x0 = self.predictor.predict_x0(pred, xt, cur_step)
        noise = self.schedule.sample_noise(xt)

        pred_noise = self.predictor.predict_noise(pred, xt, cur_step)

        sigma = (
            self.eta
            * (
                (1 - alpha_bar / alpha_bar_next)
                * (1 - alpha_bar_next)
                / (1 - alpha_bar)
            ).sqrt()
        )

        # Equation 12
        # In this equation, we use the predicted x_0, stochastic noise, and the
        # predicted noise to compute x_{t-1}.
        xt = (
            x0 * alpha_bar_next_sqred
            + sigma * noise
            # pyre-fixme
            + (((1 - alpha_bar_next) - torch.square(sigma)).sqrt()) * pred_noise
        ).to(dtype=xt.dtype)

        return xt

    def generator(
        self,
        xt: Tensor,
        c: Optional[Dict[str, Tensor]] = None,
    ) -> Generator[Tensor, None, None]:
        """Generate xt for each t in self.eval_steps"""
        steps = self.eval_steps.flip(0)
        for step, next_step in zip(steps[:-1], steps[1:]):
            # Convert steps to batched tensors
            t = step * torch.ones(xt.size(0), device=xt.device, dtype=torch.long)
            t1 = next_step * torch.ones(xt.size(0), device=xt.device, dtype=torch.long)
            # Remove noise between step t and t+1
            xt = self.remove_noise(xt, c, t, t1)
            yield xt

    def forward(
        self,
        xt: Tensor,
        timestep: Optional[Tensor] = None,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> Union[DiffusionOutput, Tensor]:
        if self.training:
            if timestep is None:
                raise ValueError("Must provide a timestep value during training")
            return self.model(xt, timestep, conditional_inputs)
        else:
            gen: Iterable = self.generator(xt, conditional_inputs)
            if self.progress_bar:
                # Lazy import so that we don't depend on tqdm.
                from tqdm.auto import tqdm

                gen = tqdm(gen, total=len(self.eval_steps) - 1)
            for x in gen:
                pass
            # pyre-ignore
            return x
