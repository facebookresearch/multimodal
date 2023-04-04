# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
from torch import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.models.clip.model import clip_vit_b32
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
from torchmultimodal.transforms.clip_transform import CLIPTransform
from torchvision.datasets import CocoCaptions
from tqdm import tqdm

bsz = 256
max_steps = 5
use_amp = True
epochs = 100


class TestDataset(Dataset):
    def __init__(self, len=5000, transform=None):
        super().__init__()
        self.len = len
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        batch = {
            "image": torch.ones(3, 224, 224),
            "text": torch.ones(77, dtype=torch.long),
        }
        if self.transform:
            image, text = self.transform(batch["image"], batch["text"])
            batch = (image, text)

        return batch["image"], batch["text"]


class COCOTransform:
    def __init__(self, clip_transform):
        self.clip_transform = clip_transform

    def __call__(self, image, target):
        text = target[0]
        return self.clip_transform(image, text)


def train():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )
    device = int(os.environ["LOCAL_RANK"])
    print("my device ", device)

    writer = SummaryWriter()
    transform = COCOTransform(CLIPTransform())
    dataset = CocoCaptions(
        root="/datasets01/COCO/022719/val2017",
        annFile="/datasets01/COCO/022719/annotations/captions_val2017.json",
        transforms=transform,
    )
    # dataset = TestDataset()
    dataloader = DataLoader(
        dataset, batch_size=bsz, sampler=DistributedSampler(dataset), num_workers=10
    )
    model = clip_vit_b32(pretrained=False)
    model = torch.compile(model)
    loss_fn = ContrastiveLossWithTemperature()
    loss_fn.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=5.0e-4,
        weight_decay=1.0e-4,
        eps=1.0e-6,
        # foreach=True
    )

    model.train()
    model.to(device)
    ddp_model = DistributedDataParallel(model, [device])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    for _ in range(epochs):
        for idx, batch in tqdm(enumerate(dataloader)):
            # print("my batch ", batch[0].size(), batch[1].size())
            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                image_embeddings, text_embeddings = ddp_model(
                    batch[0].to(device), batch[1].to(device)
                )
                loss = loss_fn(image_embeddings, text_embeddings)

            step += 1
            writer.add_scalar("train/loss", loss, step)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if idx == max_steps - 1:
            #     break

    writer.flush()
    writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
