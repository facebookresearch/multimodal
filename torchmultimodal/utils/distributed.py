# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import Tensor
from torch.distributed import all_gather as all_gather_no_backprop
from torch.distributed.nn.functional import all_gather as all_gather_with_backprop


def gather_tensor(
    tensor: Tensor, backprop_in_gather: bool = True, enable_local_backprop: bool = True
) -> List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_in_gather (bool): Whether to backpropagate the gradients from
            all_gather to all workers (versus just the local worker). Defaults
            to {\\double back-quote}True{\\double quote}.
        enable_local_backprop (bool): Whether to backpropagate the local gradients when backprop_in_gather
            is turned off. Defaults to True. It's used for cases where backpropagate should be turned off completely,
            such as metrics computation and gather constant tensor.

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()

    # This uses the all_gather from torch.distributed.nn.functional,
    # which backpropagates gradients to all workers
    if backprop_in_gather:
        return all_gather_with_backprop(tensor)

    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        # just backprop to the current worker
        # This means that the image gradients on a given worker will only
        # consider the text samples from the same worker
        if enable_local_backprop:
            tensor_all_gpus[get_rank()] = tensor
        return tensor_all_gpus


def concat_gather_all_gpu(
    tensor: Tensor,
    backprop_in_gather: bool = True,
    enable_local_backprop: bool = False,
    dim: int = 0,
) -> Tensor:
    """Gathers a tensor across all GPUs.

    Inputs:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_in_gather (bool): Whether to backpropagate the gradients from
            all_gather to all workers (versus just the local worker). Defaults
            to True.
        enable_local_backprop (bool): Whether to backpropagate the local gradients
            when backprop_in_gather is turned off.
        dim: the dimension over which the tensors are concatenated, default to 0.

    Returns:
        Tensor: concatenated gathered tensors across all GPUs.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor

    tensors_all_gpus = gather_tensor(tensor, backprop_in_gather, enable_local_backprop)

    return torch.cat(tensors_all_gpus, dim=dim)


def get_rank() -> int:
    """get rank util for distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
