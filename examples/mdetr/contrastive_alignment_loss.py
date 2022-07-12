# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from torch import Tensor
from transformers.tokenization_utils_base import BatchEncoding


def contrastive_alignment_loss(
    projected_queries: Tensor,
    projected_tokens: Tensor,
    target_tokens: List[List[List[int]]],
    indices: List[Tuple[Tensor, Tensor]],
    num_boxes: int,
    tokenized: BatchEncoding,
    temperature: float = 0.07,
) -> Tensor:
    """Contrastive alignment loss.

    Enforces alignment between the text representations after cross encoder and the
    object representations after the decoder.

                projected_queries (Tensor): Tensor containing object representations
                    projected to query dimension.
                    Size: (batch_size, num_queries, contrastive_dim)
                projected_tokens: Tensor containing text representations projected
                    to token dimension.
                    Size: (batch_size, num_tokens, contrastive_dim)
                target_tokens (List[List[List[int]]]): A very nested list of tokens
                    that correspond to each target. From outermost to innermost:
                    batch, object, list of disjoint (start, end) tokens
                indices (List[Tuple[Tensor, Tensor]]): A list of size batch_size,
                containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            num_boxes (int): Normalization factor. Should equal the average number of
                boxes per local batch.
            tokenized (BatchEncoding): Tokenized output from a transformers fast tokenizer.
                Used for token lookup based on character positions.
            temperature (float): Scaling factor used in calculating the logits.
                Default: 0.07
    """

    logits = (
        torch.matmul(projected_queries, projected_tokens.transpose(-1, -2))
        / temperature
    )  # BS x (num_queries) x (num_tokens)

    positive_map = construct_positive_map(logits, target_tokens, indices, tokenized)

    positive_logits = -logits.masked_fill(~positive_map, 0)
    negative_logits = logits

    # Calculate the contrastive loss for all objects
    boxes_with_pos = positive_map.any(2)
    pos_term = positive_logits.sum(2)
    neg_term = negative_logits.logsumexp(2)
    nb_pos = positive_map.sum(2) + 1e-6
    box_to_token_loss = (
        ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()
    )

    # Calculate the contrastive loss for all tokens
    tokens_with_pos = positive_map.any(1)
    pos_term = positive_logits.sum(1)
    neg_term = negative_logits.logsumexp(1)
    nb_pos = positive_map.sum(1) + 1e-6
    tokens_to_boxes_loss = (
        ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
    )

    tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

    return tot_loss / num_boxes


def construct_positive_map(
    logits: Tensor,
    target_tokens: List[List[List[int]]],
    indices: List[Tuple[Tensor, Tensor]],
    tokenized: BatchEncoding,
):
    # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
    # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
    positive_map = torch.zeros(logits.shape, dtype=torch.bool)
    for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, target_tokens)):
        cur_tokens = [tgt[j] for j in idx_tgt]
        for j, tok_list in enumerate(cur_tokens):
            for (beg, end) in tok_list:
                beg_pos = tokenized.char_to_token(i, beg)
                end_pos = tokenized.char_to_token(i, end - 1)

                if beg_pos is None and end_pos is None:
                    raise ValueError(
                        "At least one of beg_pos and end_pos must not be None"
                    )
                positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)
    return positive_map.to(logits.device)
