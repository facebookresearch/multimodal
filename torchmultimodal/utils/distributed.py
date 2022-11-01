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


def gather_tensor(tensor: Tensor, backprop_in_gather: bool = True) -> List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_in_gather (bool): Whether to backpropagate the gradients from
            all_gather to all workers (versus just the local worker). Defaults
            to {\\double back-quote}True{\\double quote}.

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()

    # This uses the all_gather from torch.distributed.nn.functional,
    # which backpropagates gradients to all workers
    if backprop_in_gather:
        return all_gather_with_backprop(tensor)

    # Otherwise just backprop to the current worker
    # This means that the image gradients on a given worker will only
    # consider the text samples from the same worker
    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        tensor_all_gpus[torch.distributed.get_rank()] = tensor
        return tensor_all_gpus
