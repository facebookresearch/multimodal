# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional, Union

import torch
from torch import nn, Tensor

from torchmultimodal.utils.diffusion_utils import DiffusionOutput


class CFGuidance(nn.Module):
    """
    Classifier free guidance gives diffusion models the ability to sample from a conditional
    distribution, while maintaining a healthy ratio between exploitation (i.e. correlation
    with conditional variables) and exploration (i.e. diversity of generation).
    As described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided
    Diffusion Models" (https://arxiv.org/abs/2112.10741)

    During training, `p` controls how often does the model see/use the
    unconditional embeddings (i.e. zero or random).
    During inference, `guidance` guides what ratio of unconditional vs
    conditional embeddings guide the generative output.

    Attributes:
        model (nn.Module): the neural network
        dim_cond (Dict[str, int]): the dimensions of conditional embeddings as a dictionary
            #TODO embedding sizes to be infered by running a forward pass (like LazyLinear)
        p (Union[float, Dict[str, float]]): probability values control
            conditional and unconditional generation during training.
            Defaults to 0.1 for all dim_cond keys.
        guidance (float): a magnitude to control the strength of the context_embedding
            during inference. Higher values give better alignment with the context at the
            cost of lesser diversity. Expecting a value from 0 to inf, defaults to 0.
        learn_null_emb (bool): If False, then unconditional embeddings are set to zero and are not trainable
            If True, then unconditional embeddings are set to random and are trainable. Defaults to True.
    Args:
        x (Tensor): input Tensor of shape [b, in_channels, ...]
        timestep (Tensor): diffusion step
        conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
            Conditional embeddings must have at least 2 dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        dim_cond: Dict[str, int],
        p: Union[int, float, Dict[str, float]] = 0.1,
        guidance: float = 0.0,
        learn_null_emb: bool = True,
    ):
        super().__init__()
        self.model = model
        self.dim_cond = dim_cond
        self.p = self._get_prob_dict(p)
        self.guidance = guidance
        self.learn_null_emb = learn_null_emb

        init_fn: Callable  # [[str, Tensor, Tensor], Tensor]
        if self.learn_null_emb:
            init_fn = torch.rand
            requires_grad = True
        else:
            init_fn = torch.zeros
            requires_grad = False

        self.unconditional_embedding = nn.ParameterDict(
            {
                k: nn.Parameter(init_fn(1, dim), requires_grad=requires_grad)
                for k, dim in self.dim_cond.items()
            }
        )

    def _get_prob_dict(self, prob: Union[float, Dict[str, float]]) -> Dict[str, float]:
        """
        Converting probability parameters to a consistent dict format.
        If prob is a float value, then it is assumed to be the same value for all keys
            in self.dim_cond and converted to a dict accordingly
        If prob is a dictionary, it is verfied that the keys in the probs match
            the keys in self.dim_cond

        Args:
            prob (Union[float, Dict[str, float]]): probability parameters

        """
        if isinstance(prob, float) or isinstance(prob, int):
            return {k: prob for k in self.dim_cond}
        elif isinstance(prob, dict):
            if prob.keys() != self.dim_cond.keys():
                prob_keys = ("'{}'," * (len(prob) - 1) + "'{}'").format(*prob.keys())
                dim_cond_keys = ("'{}'," * (len(self.dim_cond) - 1) + "'{}'").format(
                    *self.dim_cond.keys()
                )
                raise ValueError(
                    """prob has keys: [{}], dim_cond has keys: [{}];
                expected prob and dim_cond to have the same keys""".format(
                        prob_keys, dim_cond_keys
                    )
                )
            return prob
        else:
            raise TypeError(
                "prob should be either a float or a dict. Instead, it is a {}".format(
                    type(prob)
                )
            )

    def _update_conditions(
        self, inputs: Dict[str, Tensor], merge_func: Optional[Callable], batch_size: int
    ) -> Dict[str, Tensor]:
        """
        Merge provided conditions with learned "unconditional" embeddings. It is assumed that inputs
        are a subset of the learned unconditional embeddings. This function updates the conditional embedding
        with provided input conditions as defined by the merge_func.

        Args:
            inputs (Dict[str, Tensor]): dictionary of user provided conditional embeddings
            merge_func(Callable): function defining how to merge the conditional and unconditional embedding. This
                function takes as input the embedding key (str), and the conditional (Tensor) and unconditional
                (Tensor) embedding
            batch_size (int): batch size for the output embeddings
        """
        if not inputs.keys() <= self.dim_cond.keys():
            raise ValueError(
                "All conditional_inputs must be specified in the dim_cond dict while initiating class {}".format(
                    self.__class__.__name__
                )
            )
        embedding = dict()
        for k, uncond in self.unconditional_embedding.items():
            if k in inputs:
                cond = inputs[k]
                embedding[k] = merge_func(k, cond, uncond) if merge_func else cond
            else:
                embedding[k] = uncond.expand(batch_size, *uncond.shape[1:])
        return embedding

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> DiffusionOutput:
        conditional_inputs = conditional_inputs or {}
        b = x.shape[0]
        if self.training:
            # Classifier free guidance during training
            # Dropout randomly drops out conditional inputs based on self.p for learned unconditional ones
            dropout_func = lambda key, cond, uncond: torch.where(
                torch.rand(b, 1, device=x.device) < self.p[key], uncond, cond
            )
            embedding = self._update_conditions(conditional_inputs, dropout_func, b)
            return self.model(x, timestep, embedding)
        elif self.guidance == 0 or not conditional_inputs:
            # If guidance is 0 or there are no conditional inputs to guide, then run inference
            # with no guidance. We still update conditions incase there are conditional inputs
            # and guidance is set to 0.
            embedding = self._update_conditions(conditional_inputs, None, b)
            return self.model(x, timestep, embedding)
        else:
            # Classifier free guidance during inference
            # Cat concatenates the conditional and unconditional input to compute both model outputs
            cat_func = lambda key, cond, uncond: torch.cat(
                [cond, uncond.expand_as(cond)], dim=0
            )
            embedding = self._update_conditions(conditional_inputs, cat_func, 2 * b)

            # Duplicate x and t to perform both conditional and unconditional generation
            x_ = torch.cat([x, x], dim=0)
            t_ = torch.cat([timestep, timestep], dim=0)
            output = self.model(x_, t_, embedding)
            cond, uncond = torch.chunk(output.prediction, 2, dim=0)
            # Combine conditional and unconditional generation
            output.prediction = (1 + self.guidance) * cond - self.guidance * uncond

            # variance_value is duplicated, so deduplicating
            if output.variance_value is not None:
                output.variance_value, _ = torch.chunk(output.variance_value, 2, dim=0)
            return output
