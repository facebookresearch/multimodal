# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torchvision
from datasets import load_dataset, concatenate_datasets
from datasets.utils.file_utils import get_datasets_user_agent
from PIL import Image, UnidentifiedImageError
from pytorch_lightning import LightningDataModule
from torchvision.datasets import ImageFolder
from transformers import (
    BertTokenizer,
    DefaultDataCollator,
    DataCollatorForLanguageModeling,
    TRANSFORMERS_CACHE,
)
from transforms import (
    RandomResizedCropAndInterpolationWithTwoPic,
    MaskingGenerator,
    map_pixels,
    convert_to_rgb,
)


PRETRAINING_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
PRETRAINING_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class MaskedImageModelingTransform:
    def __init__(
        self,
        encoder_input_size: int = 224,
        codebook_input_size: int = 112,
        scale: Tuple[float, float] = (0.9, 1.0),
        encoder_interpolation: str = "bicubic",
        codebook_interpolation: str = "lanczos",
        image_mean: Tuple[float, float, float] = PRETRAINING_IMAGE_MEAN,
        image_std: Tuple[float, float, float] = PRETRAINING_IMAGE_STD,
        mask_window_size: int = 14,
        mask_num_patches: int = 75,
        mask_max_patches: Optional[int] = None,
        mask_min_patches: int = 16,
    ):
        self.common_transform = RandomResizedCropAndInterpolationWithTwoPic(
            size=encoder_input_size,
            second_size=codebook_input_size,
            scale=scale,
            interpolation=encoder_interpolation,
            second_interpolation=codebook_interpolation,
        )

        self.patch_transform = torchvision.transforms.Compose(
            [
                convert_to_rgb,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=torch.tensor(image_mean),
                    std=torch.tensor(image_std),
                ),
            ]
        )

        self.visual_token_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                map_pixels,
            ]
        )
        self.masked_position_generator = MaskingGenerator(
            mask_window_size,
            num_masking_patches=mask_num_patches,
            max_num_patches=mask_max_patches,
            min_num_patches=mask_min_patches,
        )

    def transform(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return {
            "image": self.patch_transform(for_patches),
            "image_for_codebook": self.visual_token_transform(for_visual_tokens),
            "image_patches_mask": torch.from_numpy(self.masked_position_generator()),
        }

    def __call__(self, images: Union[List[Image.Image], Image.Image]):
        if isinstance(images, list):
            output = {}
            for image in images:
                transformed_output = self.transform(image)
                for key in transformed_output:
                    if key not in output:
                        output[key] = []
                    output[key].append(transformed_output[key])
            return output
        else:
            return self.transform(images)


def default_image_pretraining_transforms():
    return MaskedImageModelingTransform(), MaskedImageModelingTransform()


def pad_batch(batch, batch_size):
    for item in batch.keys():
        if isinstance(batch[item], torch.Tensor):
            diff = batch_size - batch[item].size(0)
            pad = batch[item][-diff:].detach().clone()
            batch[item] = torch.cat([batch[item], pad], dim=0)
    return batch


class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        train_root: str,
        val_root: str,
        transforms: Optional[Tuple[Callable, Callable]] = None,
        use_subset_sampler: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches
        self.use_subset_sampler = use_subset_sampler

        if transforms is None:
            transforms = default_image_pretraining_transforms()

        self.train_transform, self.test_transform = transforms

    def setup(self, stage=None):
        # TODO: Add instructions to generate val set folder from pytorch examples repo.
        self.train_dataset = ImageFolder(
            self.train_root, transform=self.train_transform
        )
        self.val_dataset = ImageFolder(self.val_root, transform=self.test_transform)

    def _build_train_sampler(self, dataset):
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype("int")
        sampler = torch.utils.data.SubsetRandomSampler(np.where(idxs)[0])
        return sampler

    def train_dataloader(self):
        if self.use_subset_sampler:
            sampler = self._build_train_sampler(self.train_dataset)
        else:
            sampler = None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def on_before_batch_transfer(self, batch, *args):
        batch, target = batch
        batch["target"] = target
        if batch["target"].size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch


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


def _build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
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


def _encode_text_batch(batch, tokenizer, text_column="text", *args, **kwargs):
    return tokenizer(batch[text_column], *args, **kwargs)


def _encode_text(text, tokenizer, *args, **kwargs):
    return tokenizer(text, *args, **kwargs)


class TextDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_infos: List[HFDatasetInfo],
        tokenizer: Optional[Callable] = None,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        allow_uneven_batches: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_uneven_batches = allow_uneven_batches

    def setup(self, stage=None):
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        transform = partial(
            _encode_text_batch,
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        self.train_dataset = _build_datasets_from_info(
            self.dataset_infos, split="train"
        )
        self.train_dataset.set_transform(transform)
        self.val_dataset = _build_datasets_from_info(
            self.dataset_infos, split="validation"
        )
        self.val_dataset.set_transform(transform)

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset)

    def _build_dataloader(self, dataset, drop_last=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            collate_fn=self._build_collator(),
            drop_last=drop_last,
        )

    def _build_collator(self):
        return DefaultDataCollator()

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if mask.size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        batch["text_masked"] = batch.pop("input_ids")
        return batch


class MLMDataModule(TextDataModule):
    def __init__(
        self,
        dataset_infos: List[HFDatasetInfo],
        mlm_probability: float = 0.15,
        ignore_index: int = -1,
        **kwargs: Any,
    ):
        super().__init__(dataset_infos, **kwargs)
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index

    def _build_dataloader(self, dataset, drop_last=True):
        # uneven batches can cause distributed issues,
        # drop last batch to prevent those.
        # ideally, we don't need to drop these for unimodal cases
        # but just to be safe
        return self._build_dataloader(dataset, drop_last=drop_last)

    def _build_collator(self):
        return DataCollatorForLanguageModeling(
            self.tokenizer, mlm_probability=self.mlm_probability
        )

    def on_after_batch_transfer(self, batch, *args):
        batch["text_masked"] = batch.pop("input_ids")
        batch["mlm_labels"] = batch.pop("labels")
        batch["mlm_labels"][batch["mlm_labels"] == -100] = self.ignore_index
        return batch


class VLTransform:
    def __init__(self, image_transform, text_transform):
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __call__(self, info, dataset, itm_probability):
        output = {}
        text = info["text"]
        image = info["image"]
        if itm_probability > 0:
            output["itm_labels"] = torch.ones((1), dtype=torch.long)

        if random.random() < itm_probability:
            while text == info["text"]:
                text = dataset.select([random.randint(0, len(dataset) - 1)])[0]["text"]
            output["itm_labels"] = torch.zeros((1), dtype=torch.long)

        output.update(self.image_transform(image))
        output.update(self.text_transform(text))
        return output


def default_text_transform(
    text_tokenizer: Optional[Callable] = None,
    max_text_length: int = 77,
    **kwargs: Any,
):
    if text_tokenizer is None:
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    text_transform = partial(
        _encode_text,
        tokenizer=text_tokenizer,
        padding="max_length",
        max_length=max_text_length,
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

    return text_transform


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


class VLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_infos: List[HFDatasetInfo],
        val_dataset_infos: List[HFDatasetInfo],
        text_tokenizer: Optional[Callable] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        mlm_probablity: float = 0.15,
        max_text_length: int = 77,
        batch_size: int = 32,
        num_workers: int = 4,
        ignore_index: int = -1,
        itm_probability: float = 0.1,
        allow_uneven_batches: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.train_dataset_infos = train_dataset_infos
        self.val_dataset_infos = val_dataset_infos

        if image_transforms is None:
            image_transforms = default_image_pretraining_transforms()

        self.train_image_transform, self.test_image_transform = image_transforms
        self.text_tokenizer = text_tokenizer
        self.mlm_probability = mlm_probablity
        self.max_text_length = max_text_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.itm_probability = itm_probability
        self.allow_uneven_batches = allow_uneven_batches

    def setup(self, stage=None):
        if self.text_tokenizer is None:
            self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        text_transform = default_text_transform(
            self.text_tokenizer, self.max_text_length
        )
        train_vl_transform = VLTransform(self.train_image_transform, text_transform)
        val_vl_transform = VLTransform(self.test_image_transform, text_transform)
        timeout = None
        # TODO: Remove shardings in final version

        train_dataset = _build_datasets_from_info(
            self.train_dataset_infos, split="train"
        ).shard(600, 0)
        train_dataset = train_dataset.map(
            fetch_images, fn_kwargs={"timeout": timeout}, num_proc=80
        )
        train_dataset = train_dataset.filter(
            lambda example: example["image_url"] != "empty"
        )
        self.train_dataset = train_dataset
        self.train_dataset.set_transform(
            partial(
                train_vl_transform,
                dataset=train_dataset.filter(lambda example: True),
                itm_probability=self.itm_probability,
            )
        )

        val_dataset = _build_datasets_from_info(
            self.val_dataset_infos, split="validation"
        ).shard(600, 0)
        val_dataset = val_dataset.map(
            fetch_images, fn_kwargs={"timeout": timeout}, num_proc=80
        )
        val_dataset = val_dataset.filter(
            lambda example: example["image_url"] != "empty"
        )
        self.val_dataset = val_dataset
        self.val_dataset.set_transform(
            partial(
                val_vl_transform,
                dataset=self.val_dataset.filter(lambda example: True),
                itm_probability=self.itm_probability,
            )
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            collate_fn=self._build_collator(),
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            drop_last=True,
        )

    def _build_collator(self):
        # TODO: Change to whole word
        return DataCollatorForLanguageModeling(
            self.text_tokenizer, mlm_probability=self.mlm_probability
        )

    def on_before_batch_transfer(self, batch, *args):
        batch.pop("token_type_ids", None)
        mask = batch.pop("attention_mask", None)
        if mask.size(0) < self.batch_size and not self.allow_uneven_batches:
            batch = pad_batch(batch, self.batch_size)
        return batch

    def on_after_batch_transfer(self, batch, *args):
        text_masked = batch.pop("input_ids")
        mlm_labels = batch.pop("labels")
        mlm_labels[mlm_labels == -100] = self.ignore_index
        text = text_masked.detach().clone()
        text[mlm_labels != -1] = mlm_labels[mlm_labels != -1]
        batch.update(
            {"mlm_labels": mlm_labels, "text": text, "text_masked": text_masked}
        )
        return batch


FINETUNING_IMAGE_MEAN = (0.485, 0.456, 0.406)
FINETUNING_IMAGE_STD = (0.229, 0.224, 0.225)


def default_torchvision_transforms():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=FINETUNING_IMAGE_MEAN,
                std=FINETUNING_IMAGE_STD,
            ),
        ]
    )
    return transform, transform


class TorchVisionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_info: TorchVisionDatasetInfo,
        dataset_root: Optional[str] = None,
        image_transforms: Optional[Tuple[Callable, Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__()

        if dataset_root is None:
            dataset_root = os.path.join(TRANSFORMERS_CACHE, "datasets", "torchvision")
            dataset_root = os.path.join(
                dataset_root, dataset_info.class_ptr.__name__.lower()
            )
            os.makedirs(dataset_root, exist_ok=True)

        self.dataset_info = dataset_info
        self.dataset_root = dataset_root
        if image_transforms is None:
            image_transforms = default_torchvision_transforms()
        self.train_transform, self.test_transform = image_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.dataset_info.class_ptr(
            self.dataset_root,
            split=self.dataset_info.train_split,
            transform=self.train_transform,
            download=True,
        )

        if self.dataset_info.has_val:
            self.val_dataset = self.dataset_info.class_ptr(
                self.dataset_root,
                split=self.dataset_info.val_split,
                transform=self.test_transform,
                download=True,
            )

        self.test_dataset = self.dataset_info.class_ptr(
            self.dataset_root,
            split=self.dataset_info.test_split,
            transform=self.test_transform,
            download=True,
        )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        if self.dataset_info.has_val:
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        return self._build_dataloader(dataset)

    def test_dataloader(self):
        return self._build_dataloader(self.test_dataset)

    def _build_dataloader(self, dataset: torch.utils.data.Dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def on_before_batch_transfer(self, batch, *args):
        images, targets = batch
        batch = {"image": images, "labels": targets}
        return batch


class MultiDataLoader:
    # NOTE: Please check MMF's MultiDataLoader if you want to support
    # size proportional strategies or epoch based runs
    def __init__(
        self,
        loaders: List[torch.utils.data.DataLoader],
        sampling_func: Optional[Callable] = None,
    ):
        if loaders is None or len(loaders) == 0:
            warnings.warn(
                "Empty loaders passed into MultiDataLoader. This can have "
                "unintended consequences."
            )

        if sampling_func is None:
            sampling_func = partial(random.choice, range(len(loaders)))

        self.sampling_func = sampling_func
        self.loaders = loaders
        self.num_datasets = len(self.loaders)
        self.iterators = [None for _ in loaders]
        self.current_index = 0

    def set_samplers(self):
        self.samplers: List[torch.utils.data.Sampler] = []
        for loader in self.loaders:
            if hasattr(loader, "sampler"):
                self.samplers.append(loader.sampler)

    def __iter__(self):
        self.iterators = []

        for loader in self.loaders:
            self.iterators.append(iter(loader))

        self.change_dataloader()

        return self

    def __next__(self):
        """Calculation of next batch is performed using following logic.

        Current chosen iterator is set in the change_dataloader function
        based on the chosen iteration strategy which is called everytime
        prepare_batch is called.

        If we get the next batch from iterator without any StopIteration exception,
        we return it as it is. Otherwise, we have two cases:

        1. In some iteration strategies (example size proportional), each dataset
        needs to same number of epochs at any given time, we need to yield
        StopIteration exception when all iterators are finished. In turn, this
        will yield to __iter__ all reignite all of the iterators. The code will
        not reach __iter__ until unless all iterators are exhausted. An iteration
        strategy should specify this behavior through `should_exhaust_all_iterators`
        property

        2. In other cases of iteration strategies, epochs don't make sense.
        Think of a case of random (equal) proportional sampling for dataset x and y
        where x is half the size of y. When x will complete its 2 epochs, y will
        have only 1 epoch completed. **So please don't use max_epochs or epoch
        based training in this case as it won't be honored**. If an iterator is
        finished, we just reignite it in this case and finished iterators
        variable isn't used. This means that this case will never reach the
        __iter__ function ever again.


        Returns:
            SampleList: sample list instance from currently selected dataset
        """
        self.change_dataloader()
        try:
            next_batch = next(self.current_iterator)
        except StopIteration:
            iterator = iter(self.loaders[self.current_index])
            self.iterators[self.current_index] = iterator
            self.current_iterator = iterator
            next_batch = next(self.current_iterator)
        return {"batch": next_batch, "datamodule_index": self.current_index}

    def change_dataloader(self):
        choice = 0

        if self.num_datasets <= 1:
            self.current_index = choice
            self.current_iterator = self.iterators[self.current_index]
            return

        choice = [self.sampling_func()]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # This broadcast is probably unnecessary with lightning if everything
            # is already properly seeded. But, to be on safe side, we can still
            # do this.
            # TODO: Check if not doing this provides any speed benefits.
            torch.distributed.broadcast_object_list(choice, 0)

        self.current_index = choice[0]
        self.current_iterator = self.iterators[self.current_index]

    def seed_sampler(self, epoch: int):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for sampler in self.samplers:
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)


class MultiDataModule(LightningDataModule):
    # TODO: Add rest of the functions that should be called on child datamodules
    def __init__(
        self,
        datamodules: List[LightningDataModule],
        sampling_func: Optional[Callable] = None,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.sampling_func = sampling_func
        self.current_datamodule_idx = 0

    def setup(self, stage=None):
        for datamodule in self.datamodules:
            datamodule.setup(stage)

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def train_dataloader(self) -> MultiDataLoader:
        self.train_dataloader = self._build_multi_dataloader("train")
        return self.train_dataloader

    def val_dataloader(self) -> MultiDataLoader:
        self.val_dataloader = self._build_multi_dataloader("train")
        return self.val_dataloader

    def _build_multi_dataloader(self, split="train"):
        dataloaders = []
        for datamodule in self.datamodules:
            dataloaders.append(getattr(datamodule, f"{split}_dataloader")())

        return MultiDataLoader(dataloaders, self.sampling_func)

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
