# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# train using `torchrun --master_port=1256 --nproc_per_node=4 flava/plain_train.py config=flava/configs/pretraining/debug.yaml`

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
from flava.model import FLAVAPreTrainModule, get_optimizer
from flava.utils import build_config, build_datamodule_kwargs
from pytorch_lightning import seed_everything

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.modules.layers.transformer import FLAVATransformerLayer
import torchsnapshot

# optional syntax-highlighting for console output
try:
    from rich.console import Console

    c = Console(force_terminal=True)
    print = c.log
except ImportError:
    pass

writer = None
rank = None

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

def setup_distributed_device() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = rank % 8
    torch.cuda.set_device(local_rank)
    return torch.device(local_rank) if torch.cuda.is_available() else "cpu"

def wrap_model(config, device):
    model = FLAVAPreTrainModule(
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
    return model

LOG_INTERVAL = 50
def should_log(steps):
    return steps % LOG_INTERVAL == 0

def calculate_loss(output, steps, validation=False):
    losses = output.losses

    total_loss = 0
    for key in losses:
        if losses[key] is not None:
            total_loss += losses[key]
            if should_log(steps):
                loss_reduce = losses[key].detach()
                dist.reduce(loss_reduce, dst=0)
                if rank == 0:
                    if validation:
                        mode = "validation"
                    else:
                        mode = "train"
                        writer.add_scalar(
                            f"{mode}/losses/{key}",
                            loss_reduce.item() / dist.get_world_size(),
                            steps,
                        )

    return total_loss


@record
def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    device = setup_distributed_device()

    global rank
    rank = dist.get_rank()

    datamodule = get_datamodules(config)
    datamodule.setup("fit")

    model = wrap_model(config, device)

    # TODO replace with TLC logger
    global writer
    writer = SummaryWriter(f"logs/{config.training.lightning.strategy}/{int(time.time())}")

    optimizer, scheduler = get_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
    )

    steps = -1
    epochs = -1
    while True:
        t0 = time.time()
        epochs += 1
        dataloader = datamodule.train_dataloader()
        dataloader.set_epoch(epochs)

        # prof = torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=4),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/profile'),
        #         record_shapes=True,
        #         with_stack=True)
        # prof.start()

        for i, data in enumerate(dataloader):
            model.train()
            steps += 1
            data = datamodule.on_before_batch_transfer(data, None)
            data = move_to_device(data, device)
            data = datamodule.on_after_batch_transfer(data, None)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            total_loss = calculate_loss(output, steps)
            total_loss.backward()

            optimizer.step()
            scheduler.step()

            t1 = time.time()

            if should_log(steps) and rank == 0:
                batch_size = config.training.batch_size * dist.get_world_size()
                writer.add_scalar("batch/time", batch_size / (t1 - t0), steps)
            t0 = t1
            if rank == 0:
                print(f"epoch: {epochs} step {steps} loss: {total_loss.detach().item():.4}")

            if should_log(steps):
                total_loss_reduce = total_loss.detach()
                dist.reduce(total_loss_reduce.detach(), dst=0)

                if rank == 0:
                    total_loss_reduce = total_loss_reduce.item() / dist.get_world_size()

                    print(f"epoch: {epochs} step {steps} loss: {total_loss_reduce:.4}")
                    writer.add_scalar(
                        "train/loss", total_loss_reduce, steps
                    )
                    writer.add_scalar("batch_size", batch_size, steps)


            if steps % 1000 != 0 or steps == 0:
                continue

            if rank == 0:
                print("evaluating")

            model.eval()
            validation_loader = datamodule.val_dataloader()
            validation_loss = 0
            for i, data in enumerate(validation_loader):
                data = datamodule.on_before_batch_transfer(data, None)
                data = move_to_device(data, device)
                data = datamodule.on_after_batch_transfer(data, None)
                with torch.no_grad():
                    output = model(data)
                    total_loss = calculate_loss(output, steps, validation=True)
                    total_loss_reduce = total_loss.detach()
                    dist.reduce(total_loss_reduce.detach(), dst=0)
                    total_loss_reduce = total_loss_reduce.item() / dist.get_world_size()
                    validation_loss += total_loss_reduce
            if rank == 0:
                print(f"step {steps} EVAL loss: {total_loss_reduce:.4}")
                writer.add_scalar(
                    "validation/loss", total_loss_reduce, steps
                )




            



            # TODO implement validation
            # TODO implement imagenet eval
            # TODO implement checkpoint saving


if __name__ == "__main__":
    main()
