# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class OmnivoreArchitecture(nn.Module):
    """Omnivore is a model that accept multiple vision modality.

    Omnivore (https://arxiv.org/abs/2201.08377) is a single model that able to do classification
    on images, videos, and single-view 3D data using the same shared parameters of the encoder.

    Args:   encoder (nn.Module): Instantiated encoder.
                See SwinTransformer3dEncoder class.
            heads (Optinal[nn.ModuleDict]): Dictionary of multiple heads for each dataset type

    Inputs: x (Tensor): 5 Dimensional batched video tensor with format of B C D H W
                where B is batch, C is channel, D is time, H is height, and W is width.
            input_type (str): The dataset type of the input, this will used to choose
                the correct head.
    """

    def __init__(self, encoder: nn.Module, heads: nn.ModuleDict):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, x: torch.Tensor, input_type: str) -> torch.Tensor:
        x = self.encoder(x)
        assert (
            input_type in self.heads
        ), f"Unsupported input_type: {input_type}, please use one of {list(self.heads.keys())}"
        x = self.heads[input_type](x)
        return x
