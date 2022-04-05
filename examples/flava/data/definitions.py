# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import torch


def _default_split_key_mapping():
    return {x: x for x in ["train", "validation", "test"]}


@dataclass
class HFDatasetInfo:
    key: str
    subset: str
    remove_columns: Optional[List[str]] = None
    rename_columns: Optional[List[Tuple[str, str]]] = None
    # TODO: Look if we can add text column option and encode transform settings here.
    split_key_mapping: Optional[Dict[str, str]] = field(
        default_factory=_default_split_key_mapping
    )


@dataclass
class TorchVisionDatasetInfo:
    key: str
    class_ptr: torch.utils.data.Dataset
    train_split: str = "train"
    val_split: str = "val"
    has_val: bool = True
    test_split: str = "test"
