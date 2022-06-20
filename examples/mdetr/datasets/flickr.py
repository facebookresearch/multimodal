# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from examples.mdetr.datasets.modulated_detection import ModulatedDetection
from transformers import RobertaTokenizerFast


def build_flickr(image_set, tokenizer, transform, args):

    img_dir = Path(args.flickr_img_path) / f"{image_set}"

    if args.GT_type == "merged":
        identifier = "mergedGT"
    elif args.GT_type == "separate":
        identifier = "separateGT"
    else:
        assert False, f"{args.GT_type} is not a valid type of annotation for flickr"

    if args.test:
        ann_file = Path(args.flickr_ann_path) / f"final_flickr_{identifier}_test.json"
    else:
        ann_file = (
            Path(args.flickr_ann_path) / f"final_flickr_{identifier}_{image_set}.json"
        )

    is_train = image_set == "train"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
    dataset = ModulatedDetection(
        img_dir,
        ann_file,
        transforms=transform,
        return_tokens=True,  # args.contrastive_align_loss,
        tokenizer=tokenizer,
        is_train=is_train,
    )
    return dataset
