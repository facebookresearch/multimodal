# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from collections import OrderedDict
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor


def get_current_device():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


def get_extended_attention_mask(attention_mask: Tensor) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Args:
        attention_mask (Tensor): Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
    Returns:
        extended_attention_mask (Tensor): extended attention mask with the same dtype as attention_mask.dtype.
    """

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(
        dtype=attention_mask.dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def shift_dim(
    x: Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
):
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

    dims = list(range(n_dims))
    del dims[src_dim]

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


def transpose_for_scores(
    num_attention_heads: int, attention_head_size: int, x: Tensor
) -> Tensor:
    x = x.unflatten(-1, (num_attention_heads, attention_head_size))
    return x.permute(0, 2, 1, 3)


class PretrainedMixin:
    def get_model_dir(self, url):
        return os.path.join(
            torch.hub.get_dir(),
            "multimodal",
            hashlib.sha256(url.encode("utf-8")).hexdigest(),
        )

    def load_model(
        self,
        pretrained_url: Optional[str],
        load_state_dict: bool = True,
        state_dict_key: Optional[str] = None,
        strict: bool = True,
    ):
        assert isinstance(
            self, torch.nn.Module
        ), "load_model can only be called on an nn.Module instance"
        if os.path.exists(pretrained_url):
            state_dict = torch.load(pretrained_url)
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_url, model_dir=self.get_model_dir(pretrained_url)
            )
        if state_dict_key:
            state_dict = state_dict[state_dict_key]

        if load_state_dict:
            self.load_state_dict(state_dict, strict=strict)
        return state_dict


class ModelOutput(OrderedDict):
    def keys(self):
        for field in fields(self):
            yield field.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        yield from self.keys()

    def values(self):
        for field in fields(self):
            yield getattr(self, field.name)

    def items(self):
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
