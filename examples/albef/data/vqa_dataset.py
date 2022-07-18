# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Callable, List, Tuple, Union

import torch

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """
    Create the dataset for VQA task.

    Args:
        ann_file (List[str]): The paths to annotation json files.
        vqa_root (str): The path to vqa data directory.
        vg_root (str): The path to vg data directory.
        image_transform (transforms): image data transform.
        question_transform (ALBEFTextTransform): text data transform for questions.
        answer_transform (ALBEFTextTransform): text data transform for answers.
        split (str): Indicates train or test. Default is train.
        answer_list (str): The path to the answers list. Required for test split.

    Dataset Outputs:
        if split is train:
            image (Tensor): Transformed image input tensor of shape (C, W, H).
            question (Tensor): Transformed question token input ids.
            answers (List[Tensor]): List of transformed answers token input ids.
            answer_weights (List[float]): List of answer weights.
                answer_weights[i] is proportional to the number of occurences of answers[i]
        if split is test:
            image (Tensor): Transformed image input tensor of shape (C, W, H).
            question (Tensor): Transformed text token input ids.
            question_id (int): The question sample id.
    """

    def __init__(
        self,
        ann_file: List[str],
        vqa_root: str,
        vg_root: str,
        image_transform: Callable[[Image.Image], Tensor],
        question_transform: Callable[[Union[List[str], str]], Tensor],
        answer_transform: Callable[[Union[List[str], str]], Tensor],
        split: str = "train",
        answer_list: str = None,
    ) -> None:
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.image_transform = image_transform
        self.question_transform = question_transform
        self.answer_transform = answer_transform
        self.split = split

        if split == "test":
            self.answer_list = json.load(open(answer_list, "r"))
            self.answer_input_ids = self.answer_transform(self.answer_list)
            self.answer_attention_mask = (self.answer_input_ids != 0).type(torch.long)

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[Tensor, Tensor, int], Tuple[Tensor, Tensor, List[Tensor], List[float]]
    ]:
        ann = self.ann[index]

        image_root = self.vqa_root if ann["dataset"] == "vqa" else self.vg_root
        image_path = os.path.join(image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)
        question = self.question_transform(ann["question"])

        if self.split == "test":
            return image, question, ann["question_id"]

        elif self.split == "train":
            if ann["dataset"] == "vqa":
                # Each VQA sample question has a list of answers (with potential repeats)
                # answer_weight[answer] = count(answer) / len(answers for the question)
                answer_weights = {}
                for answer in ann["answer"]:
                    if answer in answer_weights.keys():
                        answer_weights[answer] += 1 / len(ann["answer"])
                    else:
                        answer_weights[answer] = 1 / len(ann["answer"])

                answers = list(answer_weights.keys())
                answer_weights = list(answer_weights.values())

            elif ann["dataset"] == "vg":
                # A VG sample question has one answer so assign it a constant weight (0.5)
                answers = [ann["answer"]]
                answer_weights = [0.5]

            answers = list(self.answer_transform(answers))

            return image, question, answers, answer_weights

        else:
            raise ValueError("dataset split should be train or test")
