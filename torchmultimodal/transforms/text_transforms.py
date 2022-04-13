# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union

import torch
import torch.nn.functional as F


class PadTransform:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, texts: torch.Tensor) -> torch.Tensor:
        max_encoded_length = texts.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            texts = F.pad(texts, (0, pad_amount))
        return texts


class StrToIntTransform:
    def __init__(self):
        pass

    def __call__(
        self, l: Union[List[str], List[List[str]]]
    ) -> Union[List[int], List[List[int]]]:
        if isinstance(l[0], str):
            return [int(x) for x in l]  # type: ignore
        if isinstance(l[0], List) and isinstance(l[0][0], str):
            return [[int(x) for x in ll] for ll in l]
        else:
            raise TypeError("Input type not supported")
