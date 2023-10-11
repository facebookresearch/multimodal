# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Dict, Optional, Protocol, runtime_checkable

from torch import Tensor

from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput


@runtime_checkable
class Adapter(Protocol):
    """Adapter modules act as wrappers on the underlying denoising model. These are flexible
    and allow the base model to be augmented to perform common diffusion tasks. Since Adapters
    share the same signature as the underlying model and the Sampler class, multiple adapters
    can be stacked together.

    Example:
        denoising_model = Unet(...)
        augmented_model = Adapter2(Adapter1(denoising_model))
        model = DDIM(augmented_model, ...)

    """

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> DiffusionOutput:
        """Model forward pass

        Args:
            x (Tensor): input Tensor of shape [b, in_channels, ...]
            timestep (Tensor): diffusion step
            conditional_inputs (Dict[str, Tensor]): conditional embedding as a dictionary.
                Conditional embeddings must have at least 2 dimensions.
        """
