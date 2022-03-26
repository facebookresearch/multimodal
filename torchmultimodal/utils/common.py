# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import os


def get_current_device():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")


class PretrainedMixin:
    def load_model(self, pretrained_url: Optional[str], load_state_dict: bool=True, state_dict_key: Optional[str] = None):
        assert isinstance(self, torch.nn.Module), "load_model can only be called on an nn.Module instance"
        if os.path.exists(pretrained_url):
            state_dict = torch.load(pretrained_url)
        else:
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url)
        if state_dict_key:
            state_dict = state_dict[state_dict_key]

        if load_state_dict:
            self.load_state_dict(state_dict)
        return state_dict