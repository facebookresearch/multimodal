# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved

import json
from pathlib import Path

import torch
import torchvision

from examples.mdetr.data.transforms import ConvertCocoPolysToMask

TYPE_TO_ID = {"obj": 0, "attr": 1, "rel": 2, "global": 3, "cat": 4}


class GQADataset(torchvision.datasets.CocoDetection):
    def __init__(
        self, img_folder, ann_file, transforms, return_tokens, tokenizer, ann_folder
    ):
        super(GQADataset, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_tokens, tokenizer=tokenizer)
        with open(ann_folder / "gqa_answer2id.json", "r") as f:
            self.answer2id = json.load(f)
        with open(ann_folder / "gqa_answer2id_by_type.json", "r") as f:
            self.answer2id_by_type = json.load(f)
        self.type_to_id = TYPE_TO_ID

    def __getitem__(self, idx):
        img, target = super(GQADataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"]
        question_id = coco_img["questionId"]
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        target["questionId"] = question_id

        if coco_img["answer"] not in self.answer2id:
            answer = "unknown"
        else:
            answer = coco_img["answer"]

        target["answer"] = torch.as_tensor(self.answer2id[answer], dtype=torch.long)
        target["answer_type"] = torch.as_tensor(
            self.type_to_id[coco_img["question_type"]], dtype=torch.long
        )
        target["answer_type_mask"] = {
            f"answer_{k}": torch.BoolTensor([True])
            if coco_img["question_type"] == k
            else torch.BoolTensor([False])
            for k in self.type_to_id.keys()
        }
        target["answer_type_mask"]["answer_type"] = torch.BoolTensor([True])

        if coco_img["answer"] not in self.answer2id_by_type["answer_attr"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_attr"] = torch.as_tensor(
            self.answer2id_by_type["answer_attr"][answer]
            if coco_img["question_type"] == "attr"
            else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_global"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_global"] = torch.as_tensor(
            self.answer2id_by_type["answer_global"][answer]
            if coco_img["question_type"] == "global"
            else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_rel"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_rel"] = torch.as_tensor(
            self.answer2id_by_type["answer_rel"][answer]
            if coco_img["question_type"] == "rel"
            else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_cat"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_cat"] = torch.as_tensor(
            self.answer2id_by_type["answer_cat"][answer]
            if coco_img["question_type"] == "cat"
            else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["answer_obj"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_obj"] = torch.as_tensor(
            self.answer2id_by_type["answer_obj"][answer]
            if coco_img["question_type"] == "obj"
            else -100,
            dtype=torch.long,
        )
        return img, target


def build_gqa(image_set, tokenizer, transform, args):
    img_dir = Path(args.vg_img_path)
    assert img_dir.exists(), f"provided VG img path {img_dir} does not exist"

    assert args.gqa_split_type is not None

    if image_set == "train":
        datasets = []
        for imset in ["train", "val"]:
            ann_file = (
                Path(args.gqa_ann_path)
                / f"finetune_gqa_{imset}_{args.gqa_split_type}.json"
            )

            datasets.append(
                GQADataset(
                    img_dir,
                    ann_file,
                    transforms=transform,
                    return_tokens=True,
                    tokenizer=tokenizer,
                    ann_folder=Path(args.gqa_ann_path),
                )
            )

        return torch.utils.data.ConcatDataset(datasets)
    elif image_set == "val":
        ann_file = Path(args.gqa_ann_path) / "finetune_gqa_testdev_balanced.json"

        return GQADataset(
            img_dir,
            ann_file,
            transforms=transform,
            return_tokens=True,
            tokenizer=tokenizer,
            ann_folder=Path(args.gqa_ann_path),
        )
    elif image_set in ["test", "challenge", "testdev", "submission"]:
        ann_file = (
            Path(args.gqa_ann_path)
            / f"finetune_gqa_{image_set}_{args.gqa_split_type}.json"
        )

        return GQADataset(
            img_dir,
            ann_file,
            transforms=transform,
            return_tokens=True,
            tokenizer=tokenizer,
            ann_folder=Path(args.gqa_ann_path),
        )

    else:
        raise ValueError(f"Unknown image set {image_set}")
