# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List

import torch
from torch import Tensor
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop


class BackpropType(Enum):
    """
    How to backpropagate gradients during all-gather op. GLOBAL will backpropagate
    to all workers, LOCAL to only the current worker, and NONE will not backpropagate
    at all.
    """

    GLOBAL = 0
    LOCAL = 1
    NONE = 2


def gather_tensor(
    tensor: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
) -> List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()

    # This uses the all_gather from torch.distributed.nn.functional,
    # which backpropagates gradients to all workers
    if backprop_type == BackpropType.GLOBAL:
        return all_gather_with_backprop(tensor)

    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        # just backprop to the current worker
        # This means that the image gradients on a given worker will only
        # consider the text samples from the same worker
        if backprop_type == BackpropType.LOCAL:
            tensor_all_gpus[get_rank()] = tensor
        return tensor_all_gpus


def concat_gather_all_gpu(
    tensor: Tensor,
    backprop_type: BackpropType = BackpropType.GLOBAL,
    dim: int = 0,
) -> Tensor:
    """Gathers a tensor across all GPUs.

    Inputs:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL
        dim: the dimension over which the tensors are concatenated, default to 0.

    Returns:
        Tensor: concatenated gathered tensors across all GPUs.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    tensors_all_gpus = gather_tensor(tensor, backprop_type)

    return torch.cat(tensors_all_gpus, dim=dim)


def get_rank() -> int:
    """get rank util for distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
