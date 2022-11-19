# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
from data.retrieval_dataset import (
    ImageToTextRetrievalDataset,
    RetrievalTrainingDataset,
    TextToImageRetrievalDataset,
)
from data.transforms import (
    ALBEFTextTransform,
    testing_image_transform,
    training_image_transform,
)
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class RetrievalDataModule(LightningDataModule):
    """
    The Data Module for Retrieval task.

    Args:
        train_files (List[str]): The paths to training json files.
        test_files (List[str]): The paths to testing json files.
        image_root (str): The path to image data directory.
        batch_size (int): The sampling batch size.
        num_workers (int): The number of workers for the distributed mode.
    """

    def __init__(
        self,
        train_files: List[str],
        test_files: List[str],
        image_root: str,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_dataset = RetrievalTrainingDataset(
            train_files,
            image_root,
            training_image_transform(),
            ALBEFTextTransform(truncate=True, max_seq_len=30, add_end_token=False),
        )

        self.image_dataset = ImageToTextRetrievalDataset(
            test_files,
            image_root,
            testing_image_transform(),
        )

        self.text_dataset = TextToImageRetrievalDataset(
            test_files,
            ALBEFTextTransform(
                truncate=True,
                pad_to_max_seq_len=True,
                max_seq_len=30,
                add_end_token=False,
            ),
        )

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_sampler(
        self,
        dataset: Dataset,
        shuffle: bool,
        is_distributed: bool,
        num_tasks: int,
        global_rank: int,
    ) -> Optional[DistributedSampler]:
        # do not return a sampler if is not in distributed mode
        # a default RandomSampler is used in this case
        if not is_distributed:
            return None

        return DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )

    def train_dataloader(
        self,
        is_distributed: bool = False,
        num_tasks: int = 0,
        global_rank: int = 0,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        DataLoader Outputs:
            images (Tensor): Tensor of shape (B, C, W, H) of image inputs.
            text (Tensor): Tensor of shape (B, L) of text inputs.
            text_atts (Tensor): Tensor of shape (B, L) of text attention mask.
            idx (Tensor): Tensor of shape (B) of image identifiers.
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=retrieval_train_collate_fn,
            drop_last=drop_last,
        )

    def image_dataloader(
        self,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        DataLoader Outputs:
            images (Tensor): Tensor of shape (B, C, W, H) of image inputs.
        """
        return DataLoader(
            self.image_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=drop_last,
        )

    def text_dataloader(
        self,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        DataLoader Outputs:
            text (Tensor): Tensor of shape (B, L) of text inputs.
            text_atts (Tensor): Tensor of shape (B, L) of text attention mask.
        """
        return DataLoader(
            self.text_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=text_collate_fn,
            drop_last=drop_last,
        )


def retrieval_train_collate_fn(
    batch: List[Tuple[Tensor, Tensor, int]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    image_list = []
    text_list = []
    idx_list = []
    for image, text, idx in batch:
        image_list.append(image)
        text_list.append(text)
        idx_list.append(idx)
    images = torch.stack(image_list, dim=0)
    text = pad_sequence(text_list, batch_first=True)
    text_atts = (text != 0).type(torch.long)
    idx = Tensor(idx_list).type(torch.long)
    return (
        images,
        text,
        text_atts,
        idx,
    )


def text_collate_fn(batch: List[Tensor]) -> Tuple[Tensor, Tensor]:
    text = pad_sequence(batch, batch_first=True)
    text_atts = (text != 0).type(torch.long)
    return text, text_atts
