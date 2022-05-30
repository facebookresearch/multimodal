# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from collections import OrderedDict
from dataclasses import fields
from typing import Optional

import torch
from torch import Tensor


def get_current_device():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


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
            self.load_state_dict(state_dict)
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
