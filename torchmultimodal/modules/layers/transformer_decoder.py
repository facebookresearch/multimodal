# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import typing
import warnings

from torch import nn
from torch.utils.checkpoint import checkpoint


@typing.no_type_check
def checkpoint_wrapper(fn):
    """Decorator to render an nn.Module method in checkpointing mode to save memory for training"""

    def inner(cls: nn.Module, *inputs, **kwargs):
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

            def create_custom_forward(fn: nn.Module):
                # Specifies what should the checkpoint API run in forward pass
                def custom_forward(*inputs):
                    return fn(cls, *inputs, **kwargs)

                return custom_forward

            return checkpoint(create_custom_forward(fn), *inputs)

        else:
            return fn(cls, *inputs, **kwargs)

    return inner
