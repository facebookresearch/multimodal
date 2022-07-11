# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
from examples.albef.data.transforms import ALBEFTransform
from examples.albef.data.vqa_dataset import VQADataset
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler


class VQADataModule(LightningDataModule):
    """
    The Data Module for Visual Question Answering task.

    Args:
        train_files (List[str]): The paths to training json files.
        test_files (List[str]): The paths to testing json files.
        answer_list (str): The path to the answers list.
        vqa_root (str): The path to vqa data directory.
        vg_root (str): The path to vg data directory.
    """

    def __init__(
        self,
        train_files: List[str],
        test_files: List[str],
        answer_list: str,
        vqa_root: str,
        vg_root: str,
    ) -> None:
        super().__init__()
        self.train_dataset = VQADataset(
            train_files,
            vqa_root,
            vg_root,
            ALBEFTransform(),
            split="train",
        )

        self.test_dataset = VQADataset(
            test_files,
            vqa_root,
            vg_root,
            ALBEFTransform(is_train=False),
            split="test",
            answer_list=answer_list,
        )

    def _get_sampler(
        self,
        dataset: VQADataset,
        shuffle: bool = False,
        is_distributed: bool = False,
        num_tasks: int = 1,
        global_rank: int = 1,
    ) -> Union[None, DistributedSampler]:
        if not is_distributed:
            return None

        return DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )

    def train_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        is_distributed: bool = False,
        num_tasks: int = 0,
        global_rank: int = 0,
    ) -> DataLoader:
        """
        DataLoader Outputs:
            images (Tensor): Tensor of shape (B, C, W, H) of image inputs.
            questions (Tensor): Tensor of shape (B, L) of question inputs.
            question_atts (Tensor): Tensor of shape (B, L) of question attention mask.
            answers (Tensor): Tensor of shape (N, M) of answer inputs.
                N >= B because a vqa sample can have multiple answers.
            answer_atts (Tensor): Tensor of shape (N, M) of answer attention mask.
            weights (Tensor): Tensor of shape (N) of answer weights.
            ans_lengths (List[int]): List of length B and sum N where
                ans_lengths[i] = number of answers for images[i] and questions[i].
        """
        sampler = self._get_sampler(
            dataset=self.train_dataset,
            shuffle=True,
            is_distributed=is_distributed,
            num_tasks=num_tasks,
            global_rank=global_rank,
        )
        shuffle = sampler is None
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=vqa_train_collate_fn,
            drop_last=True,
        )

    def test_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        is_distributed: bool = False,
        num_tasks: int = 0,
        global_rank: int = 0,
    ) -> DataLoader:
        """
        DataLoader Outputs:
            images (Tensor): Tensor of shape (B, C, W, H) of image inputs.
            questions (Tensor): Tensor of shape (B, L) of question inputs.
            question_atts (Tensor): Tensor of shape (B, L) of question attention mask.
            question_ids (List): List of length B of question ids.
        """
        sampler = self._get_sampler(
            dataset=self.test_dataset,
            shuffle=False,
            is_distributed=is_distributed,
            num_tasks=num_tasks,
            global_rank=global_rank,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=vqa_test_collate_fn,
            drop_last=False,
        )


def vqa_train_collate_fn(
    batch: List[Tuple[Tensor, Tensor, List[Tensor], List[float]]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, List[int]]:
    image_list = []
    question_list = []
    answer_list = []
    weight_list = []
    ans_lengths = []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list += answer
        weight_list += weights
        ans_lengths.append(len(answer))
    images = torch.stack(image_list, dim=0)
    questions = pad_sequence(question_list, batch_first=True)
    question_atts = (questions != 0).type(torch.long)
    answers = pad_sequence(answer_list, batch_first=True)
    answer_atts = (answers != 0).type(torch.long)
    weights = torch.Tensor(weight_list)
    return (
        images,
        questions,
        question_atts,
        answers,
        answer_atts,
        weights,
        ans_lengths,
    )


def vqa_test_collate_fn(
    batch: List[Tuple[Tensor, Tensor, int]]
) -> Tuple[Tensor, Tensor, Tensor, List[int]]:
    image_list, question_list, question_ids = [], [], []
    for image, question, question_id in batch:
        image_list.append(image)
        question_list.append(question)
        question_ids.append(question_id)
    images = torch.stack(image_list, dim=0)
    questions = pad_sequence(question_list, batch_first=True)
    question_atts = (questions != 0).type(torch.long)
    return (
        images,
        questions,
        question_atts,
        question_ids,
    )
