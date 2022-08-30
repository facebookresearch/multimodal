# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor


def get_extended_attention_mask(attention_mask: Tensor) -> Tensor:
    """Makes attention masks broadcastable along head and sequence dimensions.

    Accepting two types of attention masks:
        - Causal: masks that prevent attending to future positions of dimensions
            ``(batch_size, query_seq_len, key_seq_len)``
        - Padding: masks that prevent attending to token paddings of dimensions
            ``(batch_size, seq_len)``

    Args:
        attention_mask (Tensor):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

    Returns:
        extended_attention_mask (Tensor):
            The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
    """
    if attention_mask.dim() == 4:
        # Mask has already been broadcasted to the correct shape (either
        # [batch_size, num_heads, query_seq_length, key_seq_length] for causal case or
        # [batch_size, num_heads, seq_length, seq_length] for padding case)
        extended_attention_mask = attention_mask
    elif attention_mask.dim() == 3:
        # We can provide a self-attention mask of dimensions [batch_size, query_seq_length, key_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads,
        # [batch_size, num_heads, query_seq_length, key_seq_length].
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
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
