# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

from torch import nn

from torch.optim.lr_scheduler import LambdaLR, LRScheduler, SequentialLR
from torch.optim.optimizer import Optimizer

from torchmultimodal.modules.encoders.vision_transformer import VisionTransformer


class CosineDecay(LRScheduler):
    """
    Similar to https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    but modified to always calculate the current lr based on initial lr (vs lr from previous step) to
    ensure layer scaling happens properly in CosineWithWarmupAndLRScaling

    Args:
        optimizer (Optimizer): optimizer object
        t_max (int): maximum number of iterations
        eta_min (float): minimum learning rate. Default to 0
        last_epoch (int): the index of last epoch. Default to -1
    """

    def __init__(
        self,
        optimizer: Optimizer,
        t_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[Any]:  # type: ignore
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos((self.last_epoch) * math.pi / self.t_max))
            / 2
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]


class CosineWithWarmupAndLRScaling(SequentialLR):
    """
    Cosine decay with warmup and learning rate layer wise scaling.
    The final lr is multiplied by "lr_scale" in the optimizer param group.
    If "lr_scale" is missing, it defaults to 1.0.
    Args:
        optimizer (Optimizer): optimizer object
        max_iters (int): maximum number of iterations
        warmup_iters (int): number of warmup iterations
        min_lr (float): minimum learning rate for cosine decay. Default to 0
        last_epoch (int): the index of last epoch. Default to -1

    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_iters: int,
        warmup_iters: int,
        min_lr: float = 0,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            return current_step / warmup_iters

        multiplicative_lr = LambdaLR(
            optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch
        )

        cosine_lr = CosineDecay(
            optimizer, eta_min=min_lr, t_max=max_iters - warmup_iters
        )

        super().__init__(
            optimizer, [multiplicative_lr, cosine_lr], [warmup_iters], last_epoch
        )

    def step(self, epoch: Optional[int] = None) -> None:
        super().step()
        for idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = param_group.get("lr_scale", 1) * param_group["lr"]
            self._last_lr[idx] = param_group["lr"]  # type: ignore


def get_layer_id(param_name: str, num_layers: int) -> int:
    if param_name.startswith("embedding"):
        return 0

    # example : encoder.layer.0.attention_layernorm.weight
    if param_name.startswith("encoder.layer"):
        return int(param_name.split(".")[2]) + 1
    return num_layers


"""
Create param group where each layer lr is multiplied by factor of lr_decay ** (num_layers - current layer index)
Code adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
Args:
    model (VisionTransformer): vit model
    layer_decay (float): layer decay factor
    lr (float): base lr to which lr decay factor will be multiplied
    weight_decay (float): weight decay for params needing weight decay
    no_weight_decay_params (Tuple[str]): params without weight decay
Returns:
    Param groups map with key as param group identifier based on layer (ex: "no_decay_0" and "decay_0")
    and value as Dict with "params", "param_names", "weight_decay", "lr"
    and "lr_scale" (indicating value to be multiple to the lr field for each group)
"""


def get_param_groups_with_layer_decay(
    model: VisionTransformer,
    layer_decay: float,
    lr: float,
    weight_decay: float,
    no_weight_decay_params: Tuple[str, ...] = (
        "embeddings.cls_token",
        "embeddings.position_embeddings",
    ),
) -> Dict[str, Dict[str, Any]]:
    param_groups: Dict[str, Any] = {}
    # account for embedding by doing +1
    num_layers = len(model.encoder.layer) + 1
    # accound for final ln and head by doing +1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ndim check for ln and bias params
        if p.ndim == 1 or n in no_weight_decay_params:
            decay = 0.0
            prefix = "no_decay"
        else:
            decay = weight_decay
            prefix = "decay"
        layer_id = get_layer_id(param_name=n, num_layers=num_layers)
        param_group_name = f"{prefix}_{layer_id}"
        if param_group_name not in param_groups:
            param_groups[param_group_name] = {}
            param_groups[param_group_name]["params"] = []
            param_groups[param_group_name]["param_names"] = []
        param_groups[param_group_name]["params"].append(p)
        param_groups[param_group_name]["param_names"].append(n)
        param_groups[param_group_name]["weight_decay"] = decay
        param_groups[param_group_name]["lr"] = lr
        param_groups[param_group_name]["lr_scale"] = layer_scales[layer_id]

    return param_groups


"""
Create param group with no weight decay for bias and 1d params and weight decay for all other unfrozen params
Args:
    model(nn.Module): model
    weight_decay (float): weight decay for params needing weight decay
Returns:
    Relevant param groups
"""


def get_param_groups_with_weight_decay(
    model: nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    decay = []
    no_decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    param_groups = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
    return param_groups
