# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from torch import nn


class LateFusion(nn.Module):
    """A generic architecture for late fusion multimodal models.

    A late fusion model contains separate encoders for each modality,
    followed by a fusion layer and then a head module. For an example of a
    late fusion model, see the TorchMultimodal implementation of the cnn-lstm
    multimodal classifier (cnn_lstm.py)

    Args:
        encoders (ModuleDict): Dictionary mapping modalities to their respective
            encoders.

    Inputs:
        modalities (Dict[str, Tensor]): A dictionary mapping modalities to
            their tensor representations.
    """

    def __init__(
        self,
        encoders: nn.ModuleDict,
        fusion_module: nn.Module,
        head_module: nn.Module,
    ):
        super().__init__()
        # Sort encoders by key on init for consistency
        self.encoders = nn.ModuleDict({k: encoders[k] for k in sorted(encoders.keys())})
        self.fusion_module = fusion_module
        self.head_module = head_module

    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = {}
        for key, encoder in self.encoders.items():
            assert key in modalities, f"{key} missing in input"
            embeddings[key] = encoder(modalities[key])
        fused = self.fusion_module(embeddings)
        return self.head_module(fused)
