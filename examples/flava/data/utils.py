# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import requests
from datasets import load_dataset, concatenate_datasets
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image, UnidentifiedImageError

from .definitions import HFDatasetInfo


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []
    for dataset_info in dataset_infos:
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
        )
        if dataset_info.remove_columns is not None:
            current_dataset = current_dataset.remove_columns(
                dataset_info.remove_columns
            )
        if dataset_info.rename_columns is not None:
            for rename in dataset_info.rename_columns:
                current_dataset = current_dataset.rename_column(rename[0], rename[1])

        dataset_list.append(current_dataset)

    return concatenate_datasets(dataset_list)


def fetch_images(sample, timeout):
    if "image" in sample:
        return sample
    image_url = sample["image_url"]
    try:
        image = Image.open(
            requests.get(
                image_url,
                stream=True,
                headers={"user-agent": get_datasets_user_agent()},
                timeout=timeout,
            ).raw
        )
    except (requests.exceptions.ConnectionError, UnidentifiedImageError):
        image = Image.new("RGB", (256, 256), (255, 255, 255))
        sample["image_url"] = "empty"
    sample["image"] = image
    return sample
