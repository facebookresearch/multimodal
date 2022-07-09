# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from typing import List, Union

from examples.albef.data.transforms import ALBEFTransform

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class VQADataset(Dataset):
    """
    Create the dataset for VQA task

    Args:
        ann_file (List[str]): The paths to annotation json files.
        vqa_root (str): The path to vqa data directory.
        vg_root (str): The path to vg data directory.
        transform (ALBEFTransform): Image and text transforms.
        eos (str): The end-of-sentence token. Default is [SEP].
        split (str): Indicates train or test. Default is train.
        max_words (int): the word limit for text preprocessing in training. Default is 30.
        answer_list (str): the path to the answers list. Required for test split.
    """

    def __init__(
        self,
        ann_file: List[str],
        vqa_root: str,
        vg_root: str,
        transform: ALBEFTransform,
        split: str = "train",
        max_words: int = 30,
        answer_list: str = "",
    ) -> None:
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.transform = transform
        self.split = split
        self.max_words = max_words

        if split == "test":
            self.max_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, "r"))

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, index: int) -> Union[Tensor, Tensor, List[str], List[float]]:
        ann = self.ann[index]

        if ann["dataset"] == "vqa":
            image_path = os.path.join(self.vqa_root, ann["image"])
        elif ann["dataset"] == "vg":
            image_path = os.path.join(self.vg_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform.image_transform(image)

        if self.split == "test":
            question = pre_question(ann["question"], self.max_words)
            question = self.transform.text_transform(question)
            question_id = ann["question_id"]
            return image, question, question_id

        elif self.split == "train":

            question = pre_question(ann["question"], self.max_words)
            question = self.transform.text_transform(question)

            if ann["dataset"] == "vqa":

                answer_weight = {}
                for answer in ann["answer"]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann["answer"])
                    else:
                        answer_weight[answer] = 1 / len(ann["answer"])

                answers = list(
                    [
                        self.transform.text_transform(answer)
                        for answer in answer_weight.keys()
                    ]
                )
                weights = list(answer_weight.values())

            elif ann["dataset"] == "vg":
                answers = [self.transform.text_transform(ann["answer"])]
                weights = [0.5]

            return image, question, answers, weights


def pre_question(question, max_words):
    question = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            question.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
    )
    question = question.rstrip(" ")

    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_words:
        question = " ".join(question_words[:max_words])

    return question
