# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor


def get_extended_attention_mask(attention_mask: Tensor) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Args:
        attention_mask (Tensor):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

    Returns:
        extended_attention_mask (Tensor):
            The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads,
    # [batch_size, num_heads, from_seq_length, to_seq_length].
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for attention_mask (shape {})".format(attention_mask.shape)
        )

    extended_attention_mask = extended_attention_mask.to(
        dtype=attention_mask.dtype
    )  # fp16 compatibility

    return extended_attention_mask


def get_causal_attention_mask(
    tgt_seq_len: int, src_seq_len: Optional[int] = None
) -> Tensor:
    """
    Generates causal attention masks of dimensions (target_sequence_length, source_sequence_length).
    """
    if src_seq_len is None:
        src_seq_len = tgt_seq_len

    return torch.tril(torch.ones(tgt_seq_len, src_seq_len))
