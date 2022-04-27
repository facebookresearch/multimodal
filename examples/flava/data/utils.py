# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import requests
from datasets import load_dataset, concatenate_datasets
from datasets.utils.file_utils import get_datasets_user_agent
from definitions import HFDatasetInfo
from PIL import Image, UnidentifiedImageError

<<<<<<< HEAD

DATASETS_USER_AGENT = get_datasets_user_agent()

=======
>>>>>>> b2a1f6a ([feat] Add support for configuration system along with README)

DATASETS_USER_AGENT = get_datasets_user_agent()


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []
    for dataset_info in dataset_infos:
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
            **dataset_info.extra_kwargs,
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


def fetch_single_image(image_url, timeout, retries=0, sleep_timer=0):
    for _ in range(retries + 1):
        try:
            image = Image.open(
                requests.get(
                    image_url,
                    stream=True,
                    headers={"user-agent": DATASETS_USER_AGENT},
                    timeout=timeout,
                ).raw
            )
            break
        except (requests.exceptions.ConnectionError, UnidentifiedImageError):
            image = None
            time.sleep(sleep_timer)

    return image


def fetch_images(batch, num_threads, timeout=None, retries=0, sleep_timer=0):
    if "image" in batch:
        # This dataset already has "image" defined.
        return batch
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(
            executor.map(
                partial(
                    fetch_single_image,
                    timeout=timeout,
                    retries=retries,
                    sleep_timer=sleep_timer,
                ),
                batch["image_url"],
            )
        )
    return batch
