# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from torch import Tensor
from torchmultimodal.utils.attention import get_causal_attention_mask


def get_causal_mask(
    attention_mask: Tensor,
    input_shape: Tuple[int, int],
    has_query: bool = False,
) -> Tensor:
    """A causal mask in addition to the padding mask for Q-Former for generation task.
       when input seq_len is shorter than attn_mask, increasing causal_mask by prefix_seq_len with 1s;
       if query is available, apply causal self-attention mask to control query-text interaction;

    Arguments:
        attention_mask (Tensor) is a binary mask with 1 for unmasked and 0 for masked positions.
            Attention_mask has size of [batch_size, attn_seq_len]. attn_seq_len can be only seq_len for text_token
            or query_len + seq_len.
        input_shape (tuple[int, int]): indicates input shape of (batch_size, input_seq_len) from embedding output.
            If query_emb is used, input_seq_len is query_len + seq_len.
            Input shape can be different from attention_mask shape for image caption and text generation tasks.
        has_query (bool) indicating whether query is available in qformer input.

    Returns:
        causal_mask (Tensor): mask size of [bsz, attn_seq_len, attn_seq_len] with query,
            [bsz, input_seq_len, attn_seq_len] without query

    """
    device = attention_mask.device
    batch_size, seq_len = input_shape
    causal_mask = get_causal_attention_mask(seq_len).to(device)
    causal_mask = causal_mask.repeat(batch_size, 1).view(batch_size, seq_len, seq_len)
    # compare seq_len in input and attention mask
    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        if has_query:
            # if query is available, apply causal self-attention mask to control query-text interaction.
            # Allow queries attending each other but not the text tokens.
            causal_mask = torch.cat(
                [
                    torch.zeros(
                        (batch_size, prefix_seq_len, seq_len),
                        device=device,
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                dim=1,
            )  # mask size [bsz, attn_seq_len, input_seq_len]
        # increase causal_mask by prefix_seq_len with 1s to attend self-attention
        causal_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, causal_mask.shape[1], prefix_seq_len),
                    device=device,
                    dtype=causal_mask.dtype,
                ),
                causal_mask,
            ],
            dim=-1,
        )  # size of [bsz, attn_seq_len, attn_seq_len] with query, [bsz, input_seq_len, attn_seq_len] without query
    return causal_mask
