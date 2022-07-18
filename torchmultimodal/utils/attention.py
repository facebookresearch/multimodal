# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor


def get_causal_attention_mask(
    tgt_seq_len: int, src_seq_len: Optional[int] = None
) -> Tensor:
    """
    Generates causal attention masks of dimensions (target_seq_len, src_seq_len).
    """
    if src_seq_len is None:
        src_seq_len = tgt_seq_len

    return torch.tril(torch.ones(tgt_seq_len, src_seq_len))
