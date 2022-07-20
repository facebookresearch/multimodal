# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from examples.mdetr.data.transforms import ConvertCocoPolysToMask, create_positive_map
from torchvision.datasets import CocoDetection


class ModulatedDetection(CocoDetection):
    """The base dataset class for most MDETR datasets."""

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_tokens,
        tokenizer,
        is_train=False,
    ):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_tokens, tokenizer=tokenizer)
        self.is_train = is_train

    def __getitem__(self, idx):
        img, target = super(ModulatedDetection, self).__getitem__(idx)
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
