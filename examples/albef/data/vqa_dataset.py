# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from typing import List, Union

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(
        self,
        ann_file,
        transform: List[str],
        vqa_root: str,
        vg_root: str,
        eos: str = "[SEP]",
        split: str = "train",
        max_ques_words: int = 30,
        answer_list: str = "",
    ) -> None:
        self.split = split
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, "r"))

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        if split == "test":
            self.max_ques_words = 50  # do not limit question length during test
            self.answer_list = json.load(open(answer_list, "r"))

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(
        self, index: int
    ) -> Union[Tensor, Tensor, Tensor, Tensor]:  # TODO: not all are tensors
        ann = self.ann[index]

        if ann["dataset"] == "vqa":
            image_path = os.path.join(self.vqa_root, ann["image"])
        elif ann["dataset"] == "vg":
            image_path = os.path.join(self.vg_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.split == "test":
            question = pre_question(ann["question"], self.max_ques_words)
            question_id = ann["question_id"]
            return image, question, question_id

        elif self.split == "train":

            question = pre_question(ann["question"], self.max_ques_words)

            if ann["dataset"] == "vqa":

                answer_weight = {}
                for answer in ann["answer"]:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1 / len(ann["answer"])
                    else:
                        answer_weight[answer] = 1 / len(ann["answer"])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann["dataset"] == "vg":
                answers = [ann["answer"]]
                weights = [0.5]

            answers = [answer + self.eos for answer in answers]

            return image, question, answers, weights


def pre_question(question, max_ques_words):
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
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])

    return question
