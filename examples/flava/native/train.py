# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# train using `torchrun --master_port=1256 --nproc_per_node=4 flava/plain_train.py config=flava/configs/pretraining/debug.yaml`

import os
import sys
from contextlib import nullcontext
from typing import Any, Dict, Union

from omegaconf import DictConfig

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import time
from functools import partial

import datasets

import torch
import torch.distributed as dist
from common.data import MultiDataModule
from flava.data import ImageDataModule, MLMDataModule, VLDataModule
from flava.definitions import FLAVAArguments
from flava.native.model import FLAVAPreTrainModule, get_optimizer
from flava.native.utils import (
    build_config,
    enable_tf32,
    get_model_parameters,
    get_model_size_gb,
    move_to_device,
    print0,
    set_seed,
    setup_distributed_device,
)
from flava.utils import build_datamodule_kwargs
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing_wrapper,
)

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.modules.layers.transformer import FLAVATransformerLayer
from torchmultimodal.modules.losses.flava import FLAVAPretrainingLossOutput


def get_datamodules(config: FLAVAArguments) -> MultiDataModule:
    datamodules = []

    # also needed for the imagenet eval callback
    imagenet_datamodule = ImageDataModule(
        **build_datamodule_kwargs(config.datasets.image, config.training)
    )
    for dataset in config.datasets.selected:
        if dataset == "image":
            datamodules.append(imagenet_datamodule)
        elif dataset == "text":
            datamodules.append(
                MLMDataModule(
                    **build_datamodule_kwargs(config.datasets.text, config.training)
                )
            )
        elif dataset == "vl":
            datamodules.append(
                VLDataModule(
                    **build_datamodule_kwargs(config.datasets.vl, config.training)
                )
            )
        else:
            raise ValueError(f"unknown dataset: {dataset}")

    return MultiDataModule(datamodules)


@record
class Trainer:
    def __init__(self, config: DictConfig):
        if config.training.seed != -1:
            set_seed(config.training.seed)

        self.device: torch.device = setup_distributed_device()
        self.config: DictConfig = config
        self.rank: int = dist.get_rank()
        self._logger: SummaryWriter = SummaryWriter(
            f"logs/{config.training.strategy}/{int(time.time())}"
        )
        self.steps: int = -1
        self.epochs: int = -1

        self.datamodule: MultiDataModule = get_datamodules(config)
        self.datamodule.setup("fit")
        self.scaler = (
            torch.cuda.amp.GradScaler() if config.training.enable_amp else None
        )

    def log(
        self,
        name: str,
        value: Union[torch.Tensor, float, int],
        log_rank_0: bool = True,
        always_log: bool = False,
    ):
        if log_rank_0 and self.rank != 0:
            return

        if always_log or self.steps % self.config.training.log_interval == 0:
            self._logger.add_scalar(name, value, self.steps)

    def create_model(self) -> torch.nn.Module:
        model = FLAVAPreTrainModule(
            **self.config.get("model", {}),
        )
        strategy = self.config.training.strategy

        print0(
            f"before {strategy} model parameters: {get_model_parameters(model):,}, "
            f"size: {get_model_size_gb(model):.3} GB"
        )

        model = model.to(self.device)
        print0(f"move model to cuda: {torch.cuda.memory_allocated()/1024**3:.3} GB")
        if strategy == "ddp":
            # TODO do we have to do this in FSDP too?
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(
                model,
                device_ids=[self.rank],
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            print0(f"after DDP: {torch.cuda.memory_allocated()/1024**3:.3} GB")
        elif strategy == "fsdp":
            model = FSDP(
                model,
                # cpu_offload=CPUOffload(offload_params=True),
                mixed_precision=MixedPrecision(
                    # param_dtype=torch.bfloat16,  doesn't work
                    reduce_dtype=torch.bfloat16,
                    # buffer_dtype=torch.bfloat16,  doesn't work
                ),
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={FLAVATransformerLayer},
                ),
            )
            # doesn't work
            # apply_activation_checkpointing_wrapper(
            #     model, check_fn=lambda module: isinstance(module, FLAVATransformerLayer)
            # )
            print0(f"after FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

        else:
            raise ValueError(f"unknown strategy: {strategy}")

        print0(
            f"after {strategy} model parameters: {get_model_parameters(model):,}, "
            f"size: {get_model_size_gb(model):.3} GB"
        )

        return model

    def calculate_loss(
        self, output: FLAVAPretrainingLossOutput, validation=False
    ) -> torch.Tensor:
        losses = output.losses

        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                loss_reduce = losses[key].detach()
                dist.reduce(loss_reduce, dst=0)
                if validation:
                    mode = "validation"
                else:
                    mode = "train"
                    self.log(
                        f"{mode}/losses/{key}",
                        loss_reduce.item() / dist.get_world_size(),
                    )

        return total_loss

    def preprocess_data(self, data: Dict[str, Any]):
        data = self.datamodule.on_before_batch_transfer(data, None)
        data = move_to_device(data, self.device)
        return self.datamodule.on_after_batch_transfer(data, None)

    def train(self) -> None:
        model = self.create_model()
        optimizer, scheduler = get_optimizer(
            model,
            learning_rate=self.config.training.get("learning_rate"),
            adam_eps=self.config.training.get("adam_eps"),
            adam_weight_decay=self.config.training.get("adam_weight_decay"),
            adam_betas=self.config.training.get("adam_betas"),
            warmup_steps=self.config.training.get("warmup_steps"),
            max_steps=self.config.training.get("lightning.max_steps"),
        )

        while True:
            t0 = time.time()
            self.epochs += 1
            dataloader = self.datamodule.train_dataloader()
            dataloader.set_epoch(self.epochs)

            for i, data in enumerate(dataloader):
                self.steps += 1

                if self.config.training.max_steps < self.steps:
                    print0("Max steps reached, exiting")
                    return

                model.train()
                data = self.preprocess_data(data)
                print0(
                    f"after preprocess data {torch.cuda.memory_allocated()/1024**3:.3} GB"
                )

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast() if self.scaler else nullcontext():
                    output = model(data)
                    print0(
                        f"after forward pass {torch.cuda.memory_allocated()/1024**3:.3} GB"
                    )
                    self.log("fwd memory", torch.cuda.memory_allocated() / 1024**3)

                    total_loss = self.calculate_loss(output)
                    print0(f"after loss {torch.cuda.memory_allocated()/1024**3:.3} GB")

                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    print0(
                        f"after backward {torch.cuda.memory_allocated()/1024**3:.3} GB"
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    print0(
                        f"after backward {torch.cuda.memory_allocated()/1024**3:.3} GB"
                    )
                    optimizer.step()

                scheduler.step()

                print0(f"after step {torch.cuda.memory_allocated()/1024**3:.3} GB")

                t1 = time.time()
                batch_time = t1 - t0
                batch_size = config.training.batch_size * dist.get_world_size()
                items_time = batch_size / (t1 - t0)

                t0 = t1
                self.log("sec per batch", batch_time)
                self.log("items per sec", items_time)

                total_loss = total_loss.detach()
                dist.reduce(total_loss, dst=0)

                if self.rank == 0:
                    norm_total_loss = total_loss.item() / dist.get_world_size()

                    print(
                        f"epoch: {self.epochs} step {self.steps} loss: {norm_total_loss:.4}"
                    )
                    self.log("train/loss", norm_total_loss)
                    self.log("batch_size", batch_size)

                if (
                    self.steps % self.config.training.validation_steps != 0
                    or self.steps == 0
                ):
                    continue

                print0("evaluating")

                model.eval()
                validation_loader = self.datamodule.val_dataloader()
                validation_loss = torch.Tensor([0]).to(self.device)
                for i, data in enumerate(validation_loader):
                    if i == 20:
                        break
                    data = self.preprocess_data(data)
                    with torch.no_grad():
                        output = model(data)
                        total_loss = self.calculate_loss(output, validation=True)
                        validation_loss += total_loss.detach()

                print(validation_loss)
                dist.reduce(validation_loss, dst=0)
                norm_validation_loss = validation_loss.item() / dist.get_world_size()
                print0(f"step {self.steps} EVAL loss: {norm_validation_loss:.4}")
                self.log("validation/loss", norm_validation_loss, always_log=True)

                # TODO implement imagenet eval
                # TODO implement checkpoint saving


if __name__ == "__main__":
    datasets.logging.set_verbosity_error()  # too spammy

    config: FLAVAArguments = build_config()
    if config.training.enable_tf32:
        enable_tf32()

    trainer = Trainer(config)
    trainer.train()
