# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
from typing import Any

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import time
from functools import partial

import torch
import torch.distributed as dist
from common.data import MultiDataModule
from flava.data import ImageDataModule, MLMDataModule, VLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAPreTrainingLightningModule
from flava.utils import build_config, build_datamodule_kwargs
from pytorch_lightning import seed_everything

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.modules.layers.transformer import FLAVATransformerLayer

# optional syntax-highlighting for console output
try:
    from rich.console import Console

    c = Console(force_terminal=True)
    print = c.log
except ImportError:
    pass


# TODO replace with tlc.copy_data_to_device
def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, dict):
        d = {}
        for k, v in obj.items():
            d[k] = move_to_device(v, device)
        return d
    if isinstance(obj, list):
        l = []
        for v in obj:
            l.append(move_to_device(v, device))
        return l

    return obj.to(device)


def get_datamodules(config: FLAVAArguments) -> MultiDataModule:
    datamodules = []

    # also needed for the imagenet eval callback
    imagenet_datamodule = ImageDataModule(
        **build_datamodule_kwargs(config.datasets.image, config.training)
    )
    if "image" in config.datasets.selected:
        datamodules.append(imagenet_datamodule)

    if "text" in config.datasets.selected:
        mlm_datamodule = MLMDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training)
        )
        datamodules.append(mlm_datamodule)

    if "vl" in config.datasets.selected:
        vl_datamodule = VLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training)
        )
        datamodules.append(vl_datamodule)

    return MultiDataModule(datamodules)


def get_model_size_gb(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024 * 1024
    )


def get_model_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@record
def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    dist.init_process_group("nccl")

    rank = dist.get_rank()

    device = torch.device(rank) if torch.cuda.is_available() else "cpu"
    print("using device", device)
    torch.cuda.set_device(rank)

    datamodule = get_datamodules(config)
    datamodule.setup("fit")

    model = FLAVAPreTrainingLightningModule(
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    strategy = config.training.lightning.strategy
    if rank == 0:
        print(
            f"before {strategy} model parameters: {get_model_parameters(model):,}, "
            f"size: {get_model_size_gb(model):.3} GB"
        )

    if strategy == "ddp":
        # TODO do we have to do this in FSDP too?
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    elif strategy == "fsdp":
        model = FSDP(
            model.to(device),
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={FLAVATransformerLayer},
            ),
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    print(
        f"after {strategy} model parameters ({rank=}): {get_model_parameters(model):,}, "
        f"size: {get_model_size_gb(model):.3} GB"
    )
    # TODO replace with TLC logger
    writer = SummaryWriter(f"logs/{strategy}/{int(time.time())}")

    optimizers = model.module.get_optimizers(model)
    optimizer = optimizers[0][0]
    scheduler = optimizers[1][0]["scheduler"]

    steps = -1
    epochs = -1
    while True:
        t0 = time.time()
        epochs += 1
        dataloader = datamodule.train_dataloader()
        dataloader.set_epoch(epochs)
        for i, data in enumerate(dataloader):
            steps += 1
            dataloader_idx = 0
            data = datamodule.on_before_batch_transfer(data, dataloader_idx)
            data = move_to_device(data, device)
            data = datamodule.on_after_batch_transfer(data, dataloader_idx)

            optimizer.zero_grad()

            output = model(data, i)
            losses = output.losses
            
            total_loss = 0
            for key in losses:
                if losses[key] is not None:
                    total_loss += losses[key]
                    loss_reduce = losses[key].detach()
                    dist.reduce(loss_reduce, dst=0)
                    if rank == 0:
                        writer.add_scalar(f"train/losses/{key}", loss_reduce.item() / dist.get_world_size(), steps)
            total_loss.backward()

            if rank == 0:
                print(f"step {steps} loss: {total_loss.item():.4}")

            t1 = time.time()
            batch_size = config.training.batch_size * dist.get_world_size()
            writer.add_scalar("batch/time", batch_size / (t1 - t0), steps)
            t0 = t1

            total_loss_reduce = total_loss.detach()
            dist.reduce(total_loss_reduce.detach(), dst=0)
            if rank == 0:
                writer.add_scalar("loss", total_loss_reduce.item() / dist.get_world_size(), steps)
                writer.add_scalar("batch_size", batch_size, steps)

            optimizer.step()
            scheduler.step()

            # TODO implement validation
            # TODO implement imagenet eval
            # TODO implement checkpoint saving


if __name__ == "__main__":
    main()
