# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import fields
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from torchmultimodal import _PATH_MANAGER


def get_current_device() -> Union[str, torch.device]:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


def shift_dim(
    x: Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
) -> Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    # Remap negative dim
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = [i for i in range(n_dims) if i != src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def tensor_slice(x: Tensor, begin: List[int], size: List[int]) -> Tensor:
    """Slices a tensor dimension-wise.

    The input tensor is sliced along each dimension by specifying the starts and
    the increments.

    Args:
        x (Tensor): tensor to be sliced.
        begin (List[int]): list of starts corresponding to each dimension.
        size (List[int]): list of increments with respect to the starts along each dimension. Specifically,
                        ``-1`` means slicing from begin to the last element (inclusive) of that dimension.

    Returns:
        The sliced tensor.

    Raises:
        ValueError: if any of ``begin`` indices is negative
        ValueError: if any of ``size`` is less than ``-1``
    """
    if not all([b >= 0 for b in begin]):
        raise ValueError("All starting indices must be non-negative.")
    if not all([s >= -1 for s in size]):
        raise ValueError("All sizes must be either non-negative or -1.")

    size = [l - b if s == -1 else s for s, b, l in zip(size, begin, x.shape)]

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def load_module_from_url(
    model: nn.Module, url: str, strict: bool = True, progress: bool = True
) -> None:
    local_path = _PATH_MANAGER.get_local_path(url)
    if not torch.cuda.is_available():
        state_dict = torch.load(local_path, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(local_path)
    model.load_state_dict(state_dict, strict=strict)


@torch.no_grad()
def remove_grad(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


@torch.no_grad()
def momentum_update(model: nn.Module, model_m: nn.Module, momentum: float) -> None:
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        param_m.data = param_m.data * momentum + param.data * (1 - momentum)


class ModelOutput(OrderedDict):
    def keys(self) -> Any:
        for field in fields(self):
            yield field.name

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    def __iter__(self) -> Any:
        yield from self.keys()

    def values(self) -> Any:
        for field in fields(self):
            yield getattr(self, field.name)

    def items(self) -> Any:
        for field in fields(self):
            yield field.name, getattr(self, field.name)


def to_tuple_tuple(
    param: Union[int, Tuple[int, ...]], dim_tuple: int, num_tuple: int
) -> Tuple[Tuple[int, ...], ...]:
    """
    Convert single integer or single tuple to tuple of tuples.
    Used for kernel_size and strides parameters in convolutional models
    """
    if isinstance(param, int):
        param = (param,) * dim_tuple
    if isinstance(param, tuple):
        param_fixed = (param,) * num_tuple
    return param_fixed


def checkpoint_wrapper(fn: Callable) -> Callable:
    """Decorator to render an nn.Module instance method in checkpointing mode to save memory for training"""

    def inner(cls: nn.Module, *inputs: Any, **kwargs: Any) -> Tensor:
        if cls.training:
            # By default the checkpoint API stashes and restores the RNG state during each checkpointed
            # segment such that checkpointed passes making use of RNG (e.g., through dropout, batch norm)
            # have deterministic outputs as compared to non-checkpointed passes. This can incur a moderate
            # performance hit which we mitigate by checkpointing either before and after the layer that
            # requires RNG.
            if "use_cache" in kwargs and kwargs["use_cache"] is True:
                warnings.warn(
                    "Using `cache` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                kwargs["use_cache"] = False

            def create_custom_forward(fn: Callable) -> Callable:
                # checkpoint API does not accept user defined kwargs so we need to hide them
                def custom_forward(*inputs: Any) -> Callable:
                    return fn(cls, *inputs, **kwargs)

                return custom_forward

            # TODO: allow user to pass through use_reentrant
            return checkpoint(create_custom_forward(fn), *inputs, use_reentrant=True)

        else:
            return fn(cls, *inputs, **kwargs)

    return inner


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for i in range(n)])
