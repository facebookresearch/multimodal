# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from examples.mdetr.data.transforms import ConvertCocoPolysToMask, create_positive_map
from torchvision.datasets import CocoDetection


class ModulatedDetection(CocoDetection):
    """
    The base dataset class for most MDETR datasets.

    Follows the API for the COCO dataset. In addition to the usual image and captions,
    this class returns bounding boxes and their relationship to tokens in the caption
    as part of the target.
    """

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_tokens,
        tokenizer,
        is_train=False,
    ):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_tokens, tokenizer=tokenizer)
        self.is_train = is_train

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]

        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(
                tokenized, coco_img["tokens_positive_eval"]
            )
            target["nb_eval"] = len(target["positive_map_eval"])

        return img, target


def collate_fn(tokenizer, batch):
    batch = list(zip(*batch))
    final_batch = {}
    final_batch["samples"] = batch[0]
    final_batch["targets"] = batch[1]
    if "positive_map" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map"]
            batched_pos_map[
                cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]
            ] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        final_batch["positive_map"] = batched_pos_map.float()
    if "positive_map_eval" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map_eval"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map_eval"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map_eval"]
            batched_pos_map[
                cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]
            ] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        final_batch["positive_map_eval"] = batched_pos_map.float()
    if "answer_type_mask" in batch[1][0]:
        answer_types = {
            k: torch.cat([b["answer_type_mask"][k] for b in batch[1]])
            for k in batch[1][0]["answer_type_mask"].keys()
        }
        final_batch["answer_type_mask"] = answer_types

    if "answer" in batch[1][0]:
        answers = {}
        for f in batch[1][0].keys():
            if (
                "answer" not in f or f == "answer" or f == "answer_type_mask"
            ):  # We only use split_qa_heads = True
                continue
            answers[f] = torch.stack([b[f] for b in batch[1]])
        final_batch["answers"] = answers
    batch_encoding = tokenizer.batch_encode_plus(
        [v["caption"] for v in batch[1]], padding="longest", return_tensors="pt"
    ).to(batched_pos_map.device)
    final_batch["batch_encoding"] = batch_encoding._encodings
    return final_batch


def build_flickr(image_set, tokenizer, transform, args):

    img_dir = Path(args.flickr_img_path) / f"{image_set}"

    if args.GT_type == "merged":
        identifier = "mergedGT"
    elif args.GT_type == "separate":
        identifier = "separateGT"
    else:
        raise ValueError(f"{args.GT_type} is not a valid type of annotation for flickr")

    if args.test:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test.json"
    else:
        ann_file = (
            Path(args.flickr_ann_path) / f"final_flickr_{identifier}_{image_set}.json"
        )

    is_train = image_set == "train"

    dataset = ModulatedDetection(
        img_dir,
        ann_file,
        transforms=transform,
        return_tokens=True,
        tokenizer=tokenizer,
        is_train=is_train,
    )
    return dataset
