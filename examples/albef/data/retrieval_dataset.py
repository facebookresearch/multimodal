# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Callable, List, Tuple, Union

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class RetrievalTrainingDataset(Dataset):
    """
    Create the training dataset for Retrieval task.

    Args:
        ann_file (List[str]): The paths to training annotation json files.
        image_root (str): The path to image data directory.
        image_transform (Callable[[Image.Image], Tensor]): Image data transform.
        text_transform (Callable[[Union[List[str], str]], Tensor]): Text data transform.

    Dataset Outputs:
        image (Tensor): Transformed image input tensor of shape (C, H, W).
        caption (Tensor): Transformed text token input ids.
        idx (int): The unique identifier for the image.
    """

    def __init__(
        self,
        ann_file: List[str],
        image_root: str,
        image_transform: Callable[[Image.Image], Tensor],
        text_transform: Callable[[Union[List[str], str]], Tensor],
    ) -> None:
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        self.image_root = image_root
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.idx = {}  # map str image_id from dataset to int ids
        i = 0
        for ann in self.ann:
            image_id = ann["image_id"]
            if image_id not in self.idx.keys():
                self.idx[image_id] = i
                i += 1

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        caption = self.text_transform(ann["caption"])
        return image, caption, self.idx[ann["image_id"]]


class ImageToTextRetrievalDataset(Dataset):
    """
    Create the dataset for Image-to-Text Retrieval task.

    Args:
        ann_file (List[str]): The paths to annotation json files.
        image_root (str): The path to image data directory.
        image_transform (Callable[[Image.Image], Tensor]): Image data transform.

    Dataset Outputs:
        image (Tensor): Transformed image input tensor of shape (C, H, W).
    """

    def __init__(
        self,
        ann_file: List[str],
        image_root: str,
        image_transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.image_root = image_root
        self.image_transform = image_transform

        self.ann = []
        self.images = []  # paths to all images in the dataset
        self.image_to_text = {}  # map image ids to text ids for evaluation
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        text_id = 0
        for image_id, ann in enumerate(self.ann):
            self.images.append(ann["image"])
            num_text = len(ann["caption"])
            self.image_to_text[image_id] = list(range(text_id, text_id + num_text))
            text_id += num_text

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tensor:
        image_path = os.path.join(self.image_root, self.images[index])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        return image


class TextToImageRetrievalDataset(Dataset):
    """
    Create the dataset for Text-to-Image Retrieval task.

    Args:
        ann_file (List[str]): The paths to annotation json files.
        text_transform (Callable[[Union[List[str], str]], Tensor]): Text data transform.

    Dataset Outputs:
        text (Tensor): Transformed text token input ids.
    """

    def __init__(
        self,
        ann_file: List[str],
        text_transform: Callable[[Union[List[str], str]], Tensor],
    ) -> None:
        self.text_transform = text_transform

        self.ann = []
        self.text = []  # all text strings in the dataset
        self.text_to_image = {}  # map text ids to image ids for evaluation
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        text_id = 0
        for image_id, ann in enumerate(self.ann):
            for caption in ann["caption"]:
                self.text.append(caption)
                self.text_to_image[text_id] = image_id
                text_id += 1

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int) -> Tensor:
        text = self.text_transform(self.text[index])
        return text
