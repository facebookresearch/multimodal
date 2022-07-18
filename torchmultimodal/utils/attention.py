# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor


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

    extended_attention_mask = extended_attention_mask.to(
        dtype=attention_mask.dtype
    )  # fp16 compatibility

    return extended_attention_mask
