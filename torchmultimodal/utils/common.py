# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from collections import OrderedDict
from dataclasses import fields
from typing import Optional, Tuple

import torch


def get_current_device():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


def calculate_same_padding(kernel_size: Tuple[int, ...], stride: Tuple[int, ...]):
    # assumes that the input shape is divisible by stride
    total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
    pad_input = []
    for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
        pad_input.append((p // 2 + p % 2, p // 2))
    pad_input = tuple(sum(pad_input, tuple()))
    return pad_input


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
