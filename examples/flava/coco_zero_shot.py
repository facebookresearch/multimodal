# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch
from flava.data.transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
)
from torch import nn
from torch.utils.data import DataLoader
from torchmultimodal.models.flava.model import flava_model
from torchvision.datasets import CocoCaptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def compute_recall(similarity_scores: torch.Tensor, k: int = 5):
    dataset_size = similarity_scores.size(0)
    targets = torch.arange(dataset_size).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall


def transform(image, target):
    _, image_transform = default_image_pretraining_transforms()
    transformed_image = image_transform(image)
    # Take the first caption for now
    transformed_text = default_text_transform()(target[0])
    return transformed_image, transformed_text


def collator(batch):
    texts = []
    print(batch[0][0]["image"])
    images = torch.stack([x[0]["image"] for x in batch], dim=0)
    texts = torch.cat([torch.LongTensor(x[1]["input_ids"]) for x in batch], dim=0)
    return images, texts


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Path to data root directory")
    parser.add_argument("--annotations", help="Path to annotation file")
    parser.add_argument("--batch_size", default=16)

    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    dataset = CocoCaptions(
        root=args.data_root, annFile=args.annotations, transforms=transform
    )
    flava = flava_model(pretrained_model_key="flava_full")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    flava = flava.to(device)
    flava.eval()
    text_embeds = []
    image_embeds = []
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    for batch_idx, batch in enumerate(dataloader):
        logger.info(f"Batch id {batch_idx}")
        image, text = batch
        _, text_emb = flava.encode_text(text.to(device), projection=True)
        _, image_emb = flava.encode_image(image.to(device), projection=True)
        text_embeds.append(text_emb.detach().cpu())
        image_embeds.append(image_emb.detach().cpu())

    image_embeds = torch.cat(image_embeds, 0)
    text_embeds = torch.cat(text_embeds, 0)

    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)

    similarity_scores = image_embeds @ text_embeds.t()
    similarity_scores_t = similarity_scores.t()

    image_to_text_r1 = compute_recall(similarity_scores, k=1)
    image_to_text_r5 = compute_recall(similarity_scores, k=5)
    text_to_image_r1 = compute_recall(similarity_scores_t, k=1)
    text_to_image_r5 = compute_recall(similarity_scores_t, k=5)

    logger.info(f"image_to_text_recall@1 {image_to_text_r1}")
    logger.info(f"image_to_text_recall@5 {image_to_text_r5}")
    logger.info(f"text_to_image_recall@1 {text_to_image_r1}")
    logger.info(f"text_to_image_recall@5 {text_to_image_r5}")


if __name__ == "__main__":
    main()
