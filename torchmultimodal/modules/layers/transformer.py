# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Code for some of the transformers components in this file are initialized
# from their counterparts in Hugging Face Transformers library.

from typing import List, NamedTuple, Optional

from torch import Tensor


class TransformerOutput(NamedTuple):
    last_hidden_state: Optional[Tensor] = None
    pooler_output: Optional[Tensor] = None
    hidden_states: Optional[List[Tensor]] = None
    attentions: Optional[List[Tensor]] = None
    image_labels: Optional[Tensor] = None
