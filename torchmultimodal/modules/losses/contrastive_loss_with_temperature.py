# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, OrderedDict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.utils.distributed import BackpropType, gather_tensor


@dataclass
class ContrastiveLossOutput(OrderedDict):
    loss: Tensor
    logits_a: Tensor
    logits_b: Tensor
    loss_a: Tensor
    loss_b: Tensor


def _gather_embeddings_and_labels(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
) -> Tuple[Tensor, Tensor, Tensor]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        return embeddings_a, embeddings_b, labels

    embeddings_a_all_gpus = gather_tensor(embeddings_a, backprop_type)
    embeddings_b_all_gpus = gather_tensor(embeddings_b, backprop_type)
    # embeddings_a has shape [local_batch_size, embedding_dim]
    local_batch_size = embeddings_a.size(0)
    labels = local_batch_size * torch.distributed.get_rank() + torch.arange(
        local_batch_size, device=embeddings_a.device
    )

    return (
        torch.cat(embeddings_a_all_gpus),
        torch.cat(embeddings_b_all_gpus),
        labels,
    )


def contrastive_loss_with_temperature(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    logit_scale: nn.Parameter,
    mask: Optional[Tensor] = None,
    backprop_type: BackpropType = BackpropType.GLOBAL,
    cross_entropy_kwargs: Optional[Dict[str, Any]] = None,
) -> ContrastiveLossOutput:
    """Functional component for the ContrastiveLossWithTemperature. Please
    check the class for more details

    Args:
        embeddings_a (Tensor): Tensor containing features from the first input or modality.
            (In the CLIP model, these are the outputs of the image encoder.)
        embeddings_b (Tensor): Tensor containing features from the second input or modality.
            (In the CLIP model, these are the outputs of the text encoder.)
        logit_scale (nn.Parameter): Parameter with value of log of the learned temperature
        mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
            be considered in the loss calculation use this option to pass a boolean
            mask. Size is (BatchSize,). Defaults to None.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL
        cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)

    Returns:
        ContrastiveLossOutput: instance of ContrastiveLossOutput with all of the
            relevant fields.
    """

    # this temperature implementation follows CLIP Figure 3
    temperature = torch.exp(logit_scale)

    (
        embeddings_a_all_gpus,
        embeddings_b_all_gpus,
        labels,
    ) = _gather_embeddings_and_labels(embeddings_a, embeddings_b, backprop_type)

    # logits_per_image has shape [local_batch_size, global_batch_size]
    logits_per_input_a = (
        torch.matmul(embeddings_a, embeddings_b_all_gpus.transpose(0, 1)) * temperature
    )
    logits_per_input_b = (
        torch.matmul(embeddings_b, embeddings_a_all_gpus.transpose(0, 1)) * temperature
    )

    if mask is not None:
        logits_per_input_a = logits_per_input_a[mask]
        logits_per_input_b = logits_per_input_b[mask]
        labels = labels[mask]

    if cross_entropy_kwargs is None:
        cross_entropy_kwargs = {}

    loss_a = F.cross_entropy(logits_per_input_a, labels, **cross_entropy_kwargs)
    loss_b = F.cross_entropy(logits_per_input_b, labels, **cross_entropy_kwargs)
    loss = (loss_a + loss_b) / 2

    return ContrastiveLossOutput(
        loss=loss,
        logits_a=logits_per_input_a,
        logits_b=logits_per_input_b,
        loss_a=loss_a,
        loss_b=loss_b,
    )


DEFAULT_LOGIT_SCALE = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))


class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of input embeddings a and b. For each input_a
    embedding, we compute a weighted cosine similarity with all input_b embeddings,
    then calculate the cross entropy loss against the true (input_a, input_b) pairing.
    Each input_b embedding is evaluated against all input_a embeddings similarly.
    The batch's loss is the average cross entropy over all input_a and input_b embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (Union[float, nn.Module]): Log of the learnable temperature parameter value
            A nn.Parameter instantiation can also be passed directly in case parent class
            is handling the initialization.
            Defaults to ``ln(1/0.07)``, as in the CLIP paper.
        logit_scale_min (Optional[float]): Log of the minimum temperature value.
            If ``None``, then temperature will not be clamped to a minimum value.
            Defaults to ``ln(1)``, as in the CLIP paper.
        logit_scale_max (Optional[float]): Log of the maximum temperature value.
            If ``None``, then temperature will not be clamped to a maximum value.
            Defaults to ``ln(100)``, as in the CLIP paper.

    Inputs: embeddings_a (Tensor): Tensor containing features from the first input or modality.
                (In the CLIP model, these are the outputs of the image encoder.)
            embeddings_b (Tensor): Tensor containing features from the second input or modality.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_type (BackpropType): whether to backpropagate gradients to all
                workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
                Default: BackpropType.GLOBAL
            cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)
            mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
                be considered in the loss calculation use this option to pass a boolean
                mask. Size is (BatchSize,). Defaults to None.
    """

    def __init__(
        self,
        logit_scale: Union[float, nn.Parameter] = DEFAULT_LOGIT_SCALE,
        logit_scale_min: Optional[float] = math.log(1),
        logit_scale_max: Optional[float] = math.log(100),
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        if not logit_scale_min and not logit_scale_max:
            raise ValueError(
                "Only one of `logit_scale_min` and `logit_scale_max` can be None."
            )
        self.logit_scale_min = logit_scale_min
        self.logit_scale_max = logit_scale_max

        # If already initialized, set to what was passed
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        backprop_type: BackpropType = BackpropType.GLOBAL,
        cross_entropy_kwargs: Optional[Dict[str, Any]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        return contrastive_loss_with_temperature(
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            logit_scale=self.logit_scale,
            backprop_type=backprop_type,
            cross_entropy_kwargs=cross_entropy_kwargs,
            mask=mask,
        ).loss
