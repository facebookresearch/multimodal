# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor

from torchmultimodal.diffusion_labs.modules.adapters.adapter import Adapter
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


class CFGuidance(nn.Module, Adapter):
    """
    Classifier free guidance gives diffusion models the ability to sample from a conditional
    distribution, while maintaining a healthy ratio between exploitation (i.e. correlation
    with conditional variables) and exploration (i.e. diversity of generation).
    As described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided
    Diffusion Models" (https://arxiv.org/abs/2112.10741)

    During training, `p` controls how often does the model see/use the
    unconditional embeddings (i.e. zero or random or user-provided embedding).
    During inference, `guidance` guides what ratio of unconditional vs
    conditional embeddings guide the generative output. Additionally
    `eval_unconditional_embeddings` provides an option to specify an alternative
    embedding, other than the one used to train the model.

    Attributes:
        model (nn.Module): the neural network
        dim_cond (Dict[str, Union[int, Sequence[int]]]): the dimensions of conditional embeddings as
            a dictionary. Keys are names of the conditionals and values are either integers representing
            a single dimension or a sequence to represent multiple dimensions. For example, [x, y] would
            mean that the conditional embeddings are of size [batch_size, x, y].
            #TODO embedding sizes to be infered by running a forward pass (like LazyLinear)
        p (Union[float, Dict[str, float]]): probability values control
            conditional and unconditional generation during training.
            Defaults to 0.1 for all dim_cond keys.
        guidance (float): a magnitude to control the strength of the context_embedding
            during inference. Higher values give better alignment with the context at the
            cost of lesser diversity. Expecting a value from 0 to inf, defaults to 0.
        learn_null_emb (bool): If False, then unconditional embeddings are set to zero and are not trainable
            If True, then unconditional embeddings are set to random and are trainable. Defaults to True.
        train_unconditional_embeddings (Optional[Dict[str, Tensor]]): initial values to be used for
            unconditional embeddings for training. If not provided, random values or zero values are
            initialized. Defaults to None.
        eval_unconditional_embeddings (Optional[Dict[str, Tensor]]): unconditional embeddings to be used for
            evaluation. If not provided, the learned unconditional embeddings will be used. Defaults to None.
    Args:
        x (Tensor): input Tensor of shape [b, in_channels, ...]
        timestep (Tensor): diffusion step
        conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
            Conditional embeddings must have at least 2 dimensions.
    """

    def __init__(
        self,
        model: nn.Module,
        dim_cond: Dict[str, Union[int, Sequence[int]]],
        p: Union[int, float, Dict[str, float]] = 0.1,
        guidance: float = 0.0,
        learn_null_emb: bool = True,
        train_unconditional_embeddings: Optional[Dict[str, Tensor]] = None,
        eval_unconditional_embeddings: Optional[Dict[str, Tensor]] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.model = model
        self.dim_cond = dim_cond
        self.p = self._get_prob_dict(p)
        self.guidance = guidance
        self.learn_null_emb = learn_null_emb

        init_fn: Callable
        if self.learn_null_emb:
            init_fn = torch.rand
            requires_grad = True
        else:
            init_fn = torch.zeros
            requires_grad = False

        # Use user provided initial embeddings if provided, otherwise initialize with
        # zeros or random values
        self.train_unconditional_embedding = self._gen_unconditional_embeddings(
            train_unconditional_embeddings, init_fn, requires_grad
        )

        # Initialize eval embeddings with train embeddings
        # ParameterDict.copy() creates a new ParameterDict but keeps the references to
        # the same parameters as train. So any parameter updates should be avaialble here.
        self.eval_unconditional_embedding = self.train_unconditional_embedding.copy()
        # Update eval embeddings with user provided embeddings, if provided
        if eval_unconditional_embeddings is not None:
            self.eval_unconditional_embedding.update(
                {
                    key: nn.Parameter(val, requires_grad=False)
                    for key, val in eval_unconditional_embeddings.items()
                }
            )

    def _gen_unconditional_embeddings(
        self,
        initial_embeddings: Optional[Dict[str, Tensor]],
        default_init_fn: Callable,
        requires_grad: bool,
    ) -> nn.ParameterDict:
        """
        Generate parameter dictionary for unconditional embeddings based on the dim values.
        Args:
            initial_embeddings (Optional[Dict[str, Tensor]]): initial embedding values for each key,
                If embedding is not provided for a key, then use `default_init_fn` to initialize.
            default_init_fn (Callable): function to initialize an embedding
            requires_grad (bool): whether or not to optimize this embedding
        Returns:
            params (nn.ParameterDict): dictionary of unconditional embeddings
        """
        param_dict = {}
        for key, dim in self.dim_cond.items():
            if initial_embeddings and key in initial_embeddings:
                param_dict[key] = nn.Parameter(
                    initial_embeddings[key], requires_grad=requires_grad
                )
            else:
                shape = (1,) + (tuple(dim) if isinstance(dim, Sequence) else (dim,))
                param_dict[key] = nn.Parameter(
                    default_init_fn(*shape), requires_grad=requires_grad
                )
        return nn.ParameterDict(param_dict)

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

    def _extract_guidance_conditions(
        self, inputs: Optional[Dict[str, Tensor]]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Split conditional_inputs into conditions for guidance and other inputs

        Args:
           inputs (Dict[str, Tensor]): dictionary of user provided conditional embeddings
        """
        conditions, others = {}, {}
        if inputs is not None:
            for k, v in inputs.items():
                if k in self.dim_cond:
                    conditions[k] = v
                else:
                    others[k] = v
        return conditions, others

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
        embedding = dict()
        # Pick the correct unconditional embedding for train or eval
        unconditional_embedding = (
            self.train_unconditional_embedding
            if self.training
            else self.eval_unconditional_embedding
        )
        for k, uncond in unconditional_embedding.items():
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
        conditional_inputs, other_inputs = self._extract_guidance_conditions(
            conditional_inputs
        )
        b = x.shape[0]
        if self.training:
            # Classifier free guidance during training
            # Dropout randomly drops out conditional inputs based on self.p for learned unconditional ones
            dropout_func = lambda key, cond, uncond: torch.where(
                # rand tensor must have same number of dims as cond and uncond
                torch.rand(b, *([1] * (len(cond.shape) - 1)), device=x.device)
                < self.p[key],
                uncond,
                cond,
            )
            embedding = self._update_conditions(conditional_inputs, dropout_func, b)
            embedding.update(other_inputs)
            return self.model(x, timestep, embedding)
        elif self.guidance == 0 or not conditional_inputs:
            # If guidance is 0 or there are no conditional inputs to guide, then run inference
            # with no guidance. We still update conditions incase there are conditional inputs
            # and guidance is set to 0.
            embedding = self._update_conditions(conditional_inputs, None, b)
            embedding.update(other_inputs)
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
            other_ = {k: torch.cat([v, v], dim=0) for k, v in other_inputs.items()}
            embedding.update(other_)
            output = self.model(x_, t_, embedding)
            cond, uncond = torch.chunk(output.prediction, 2, dim=0)
            # Combine conditional and unconditional generation
            output.prediction = (1 + self.guidance) * cond - self.guidance * uncond

            # variance_value is duplicated, so deduplicating
            if output.variance_value is not None:
                output.variance_value, _ = torch.chunk(output.variance_value, 2, dim=0)
            return output
