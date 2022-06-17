# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import torch
from torch.utils.data import DataLoader
import torchvision
from definitions import HFDatasetInfo, TorchVisionDatasetInfo
from pytorch_lightning import LightningDataModule
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    DefaultDataCollator,
    TRANSFORMERS_CACHE,
)
from transformers.data.data_collator import torch_default_data_collator
from torchdata.datapipes.iter import Mapper, SampleMultiplexer, IterDataPipe, IterableWrapper
from .transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
    default_torchvision_transforms,
    encode_text_batch,
    pad_batch,
    TEXT_DEFAULT_TOKENIZER,
    TEXT_WHOLE_WORD_MASK_TOKENIZER,
    VL_MAX_LENGTH_DEFAULT,
    VLTransform,
)
from .utils import build_datapipes_from_info, build_datasets_from_info, fetch_images


def transform_image(transform, sample):
    sample.update(transform(sample["image"]))
    return sample


class DataCollatorForWholeWordMaskRetainingBatch(DataCollatorForWholeWordMask):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        masked_batch = super().torch_call(examples)
        examples = torch_default_data_collator(examples)
        examples["input_ids"] = masked_batch["input_ids"]
        examples["labels"] = masked_batch["labels"]
        return examples

class TextDataPipeModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        text_columns: List[str],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.train_dataset_infos = train_infos
        self.text_columns = text_columns
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
        transform = partial(
            encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            text_columns=self.text_columns,
            return_batch=True,
        )

        # previous HF dataset
        self.train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        self.train_dataset.set_transform(transform)
        self.val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )
        self.val_dataset.set_transform(transform)

        self.train_datapipe_source = IterableWrapper(self.train_dataset)
        self.val_datapipe_source = IterableWrapper(self.val_dataset)

    def train_datapipe(self):
        return self._build_datapipe(datapipe=self.train_datapipe_source,collate_fn=self._build_collator)

    def val_datapipe(self):
        return self._build_datapipe(datapipe=self.val_datapipe_source, collate_fn=self._build_collator, shuffle=False)

    def _build_datapipe(self, datapipe: IterDataPipe,  collate_fn : Any, drop_last: bool = False, shuffle: bool = True):
        output_datapipe = datapipe.batch(batch_size=self.batch_size, drop_last=drop_last).collate(collate_fn)
        if shuffle:
            return output_datapipe.shuffle()
        return output_datapipe

    def _build_collator(self, batch):
        return DefaultDataCollator().__call__(batch)

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if mask.size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        batch["text"] = batch.pop("input_ids")
        return batch


class MLMDataPipeModule(TextDataPipeModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        text_columns: List[str],
        val_infos: Optional[List[HFDatasetInfo]] = None,
        mlm_probability: float = 0.15,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__(train_infos, text_columns, val_infos, **kwargs)
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(TEXT_DEFAULT_TOKENIZER)
        transform = partial(
            encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
            text_columns=self.text_columns,
            return_batch=False,
        )

        # previous HF dataset setup
        self.train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        self.train_dataset.set_transform(transform)
        self.val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )
        self.val_dataset.set_transform(transform)
        self.train_datapipe_source = IterableWrapper(self.train_dataset)
        self.val_datapipe_source = IterableWrapper(self.val_dataset)

    def train_datapipe(self):
        return self._build_datapipe(datapipe=self.train_datapipe_source,collate_fn=self._build_collator)

    def val_datapipe(self):
        return self._build_datapipe(datapipe=self.val_datapipe_source, collate_fn=self._build_collator, shuffle=False)

    def _build_datapipe(self, datapipe: IterDataPipe, collate_fn: Any, drop_last: bool = True, shuffle: bool = True):
        # uneven batches can cause distributed issues,
        # drop last batch to prevent those.
        # ideally, we don't need to drop these for unimodal cases
        # but just to be safe
        return super()._build_datapipe(datapipe, collate_fn, drop_last=drop_last, shuffle=shuffle)

    def _build_collator(self, batch):
        return DataCollatorForLanguageModeling(
            self.tokenizer, mlm_probability=self.mlm_probability
        ).__call__(batch)

    def on_after_batch_transfer(self, batch, *args):
        batch["text_masked"] = batch.pop("input_ids")
        batch["mlm_labels"] = batch.pop("labels")
        batch["mlm_labels"][batch["mlm_labels"] == -100] = self.ignore_index
        return batch


class VLDataPipeModule(LightningDataModule):
    def __init__(
        self,
        train_infos: List[HFDatasetInfo],
        val_infos: List[HFDatasetInfo],
        text_transform: Optional[Callable] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        mlm_probablity: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        finetuning: bool = False,
        ignore_index: int = -1,
        itm_probability: float = 0.1,
        allow_uneven_batches: bool = False,
        fetch_num_threads: int = 4,
        fetch_retries: int = 0,
        fetch_sleep_timer: int = 0,
        fetch_timeout: Optional[float] = None,
        fetch_batch_size: int = 50,
        **kwargs,
    ):
        super().__init__()

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        if image_transforms is None:
            if not finetuning:
                image_transforms = default_image_pretraining_transforms()
            else:
                image_transforms = default_torchvision_transforms(use_dict=True)

        self.train_image_transform, self.test_image_transform = image_transforms
        self.text_transform = text_transform
        self.mlm_probability = mlm_probablity
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.itm_probability = itm_probability
        self.allow_uneven_batches = allow_uneven_batches
        self.fetch_num_threads = fetch_num_threads
        self.fetch_retries = fetch_retries
        self.fetch_sleep_timer = fetch_sleep_timer
        self.fetch_timeout = fetch_timeout
        self.fetch_batch_size = fetch_batch_size

    def setup(self, stage=None):
        if self.text_transform is None:
            # TODO Update to use whole word mask vocab
            text_tokenizer = BertTokenizer.from_pretrained(
                TEXT_WHOLE_WORD_MASK_TOKENIZER
            )
            self.text_transform = default_text_transform(
                text_tokenizer, max_text_length=VL_MAX_LENGTH_DEFAULT
            )
        self.text_tokenizer = self.text_transform.keywords["tokenizer"]
        self.train_vl_transform = VLTransform(
            self.train_image_transform, self.text_transform
        )
        self.val_vl_transform = VLTransform(self.test_image_transform, self.text_transform)

        train_dataset = build_datasets_from_info(
            self.train_dataset_infos, split="train"
        )
        train_dataset = train_dataset.map(
            fetch_images,
            batched=True,
            batch_size=self.fetch_batch_size,
            fn_kwargs={
                "num_threads": self.fetch_num_threads,
                "timeout": self.fetch_timeout,
                "retries": self.fetch_retries,
                "sleep_timer": self.fetch_sleep_timer,
            },
        )
        train_dataset = train_dataset.filter(
            lambda example: example["image"] is not None
        )
        self.train_dataset = train_dataset
        self.train_dataset.set_transform(
            partial(
                self.train_vl_transform,
                dataset=train_dataset.filter(lambda example: True),
                itm_probability=self.itm_probability,
            )
        )

        val_dataset = build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        )

        val_dataset = val_dataset.map(
            fetch_images,
            batched=True,
            batch_size=self.fetch_batch_size,
            fn_kwargs={
                "num_threads": self.fetch_num_threads,
                "timeout": self.fetch_timeout,
                "retries": self.fetch_retries,
                "sleep_timer": self.fetch_sleep_timer,
            },
        )
        val_dataset = val_dataset.filter(lambda example: example["image"] is not None)
        self.val_dataset = val_dataset
        self.val_dataset.set_transform(
            partial(
                self.val_vl_transform,
                dataset=self.val_dataset.filter(
                    lambda example: True
                ),  # Pass a copy to transform
                itm_probability=self.itm_probability,
            )
        )
        self.train_datapipe_source = IterableWrapper(self.train_dataset)
        self.val_datapipe_source = IterableWrapper(self.val_dataset)

    def train_datapipe(self):
        return self._build_datapipe(self.train_datapipe_source, self._build_collator, drop_last=True, shuffle=True)

    def val_datapipe(self):
        return self._build_datapipe(self.val_datapipe_source, self._build_collator, drop_last=True, shuffle=False)

    def _build_datapipe(self, datapipe: IterDataPipe,  collate_fn : Any, drop_last: bool = False, shuffle: bool = True):
        output_datapipe = datapipe.batch(batch_size=self.batch_size, drop_last=drop_last).collate(collate_fn)
        if shuffle:
            return output_datapipe.shuffle()
        return output_datapipe

    def _build_collator(self, batch):
        return DataCollatorForWholeWordMaskRetainingBatch(
            self.text_tokenizer, mlm_probability=self.mlm_probability
        ).__call__(batch)

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if (
            mask is not None
            and mask.size(0) < self.batch_size
            and not self.allow_uneven_batches
        ):
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        text_masked = batch.pop("input_ids")
        mlm_labels = batch.pop("labels", None)
        mlm_labels[mlm_labels == -100] = self.ignore_index
        text = text_masked.detach().clone()
        text[mlm_labels != -1] = mlm_labels[mlm_labels != -1]
        batch.update(
            {"mlm_labels": mlm_labels, "text": text, "text_masked": text_masked}
        )
        return batch

class MultiDataPipeModule(LightningDataModule):
    """MultiDataPipeModule is a datamodule with multiple datapipes.
    """

    # NOTE: Add rest of the functions that should be called on child datamodules
    # as required
    def __init__(
        self,
        datamodules: List[LightningDataModule],
        sampling_func: Optional[Callable] = None,
        num_workers: int = 4,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.sampling_func = sampling_func
        self.num_workers = num_workers

    def setup(self, stage=None):
        for datamodule in self.datamodules:
            datamodule.setup(stage)

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def train_dataloader(self) -> DataLoader:
        # TODO: Fix assign inconsistency
        return self._build_multi_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._build_multi_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._build_multi_dataloader("test")

    def _build_multi_dataloader(self, split="train"):
        datapipes = []
        current_index = -1
        for datamodule in self.datamodules:
            current_index += 1
            individual_datapipe = getattr(datamodule, f"{split}_datapipe")()
            individual_datapipe = Mapper(individual_datapipe, lambda x: {"batch": x, "datamodule_index":current_index})
            datapipes.append(individual_datapipe)


        if self.sampling_func is None:
            # use random sample
            # cycle short datapipes
            max_steps = self.trainer.max_steps
            datapipes = [dp.cycle(math.ceil(max_steps/len(dp))) for dp in datapipes]
            # random sample
            sample_rate = 1/len(datapipes)
            datapipe_dict = {dp:sample_rate for dp in datapipes}
            # seed value can be changed
            output_datapipe = SampleMultiplexer(datapipe_dict, seed=0)
        return DataLoader(output_datapipe, batch_size=None, num_workers=4, sampler=None)

    def on_before_batch_transfer(self, batch, *args):
        batch, index = batch["batch"], batch["datamodule_index"]
        self.current_datamodule_idx = index
        return self.datamodules[self.current_datamodule_idx].on_before_batch_transfer(
            batch, *args
        )

    def on_after_batch_transfer(self, batch, *args):
        return self.datamodules[self.current_datamodule_idx].on_after_batch_transfer(
            batch, *args
        )

    def teardown(self, stage):
        for datamodule in self.datamodules:
            datamodule.teardown(stage)
