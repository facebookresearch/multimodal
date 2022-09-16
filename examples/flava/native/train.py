# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# example command to train:
# `torchrun --nproc_per_node=8 -m flava.native.train config=flava/native/configs/pretrain_debug.yaml`


import time
from functools import partial
from typing import Any, Dict, Tuple, Union

import datasets
import numpy as np
import torch
import torch.distributed as dist
from common.data import MultiDataModule
from flava.definitions import FLAVAArguments
from flava.native.data import (
    default_text_transform,
    ImageDataModule,
    MLMDataModule,
    VL_MAX_LENGTH_DEFAULT,
    VLDataModule,
)
from flava.native.model import FLAVAPreTrainModule, get_optimizer
from flava.native.utils import (
    build_config,
    enable_tf32,
    get_model_parameters,
    get_model_size_gb,
    move_to_device,
    print0,
    run_imagenet_zero_shot,
    set_seed,
    setup_distributed_device,
)
from flava.utils import build_datamodule_kwargs

from omegaconf import DictConfig, OmegaConf
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing_wrapper,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmultimodal.models.flava.image_encoder import ImageTransformer
from torchmultimodal.models.flava.text_encoder import BERTTextEncoder
from torchmultimodal.modules.layers.transformer import TransformerEncoderLayer
from torchmultimodal.modules.losses.flava import FLAVAPretrainingLossOutput


def get_datamodules(config: FLAVAArguments) -> Tuple[MultiDataModule, ImageDataModule]:
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

    return MultiDataModule(datamodules), imagenet_datamodule


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

        multi_module, image_module = get_datamodules(config)

        self.datamodule: MultiDataModule = multi_module
        self.datamodule.setup("fit")

        self.imagenet_val_dataloader = image_module.val_dataloader()
        self.imagenet_val_text_transform = default_text_transform(
            max_text_length=VL_MAX_LENGTH_DEFAULT
        )

        self.half_dtype = (
            torch.bfloat16
            if config.training.half_precision_format == "bfloat16"
            else torch.float16
        )

        self.scaler = ShardedGradScaler() if config.training.enable_amp else None

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
        model_config = self.config.get("model", {})
        print0(f"using model config: {model_config}")

        model = FLAVAPreTrainModule(**model_config)
        strategy = self.config.training.strategy

        print0(
            f"before {strategy} model parameters: {get_model_parameters(model):,}, "
            f"size: {get_model_size_gb(model):.3} GB"
        )

        if self.config.training.activation_checkpointing:
            check_fn = lambda submodule: isinstance(submodule, TransformerEncoderLayer)
            if self.config.training.activation_checkpointing_reentrant:
                checkpoint_impl = CheckpointImpl.REENTRANT
            else:
                checkpoint_impl = CheckpointImpl.NO_REENTRANT

            non_reentrant_wrapper = partial(
                checkpoint_wrapper,
                offload_to_cpu=False,
                checkpoint_impl=checkpoint_impl,
            )
            apply_activation_checkpointing_wrapper(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

        if strategy == "ddp":
            # TODO do we have to do this in FSDP too? see https://github.com/pytorch/pytorch/issues/75478
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(self.device)

            print0(
                f"after moving to cuda: {torch.cuda.memory_allocated()/1024**3:.3} GB"
            )

            model = DDP(
                model,
                device_ids=[self.rank],
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            print0(f"after DDP: {torch.cuda.memory_allocated()/1024**3:.3} GB")
        elif strategy == "fsdp":
            mp = None
            if self.config.training.enable_half_reduce_in_fsdp:
                mp = MixedPrecision(
                    # param_dtype=self.half_dtype,  not working
                    reduce_dtype=self.half_dtype,
                    # buffer_dtype=self.half_dtype,
                )

            model = FSDP(
                model,
                mixed_precision=mp,
                device_id=self.device,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        TransformerEncoderLayer,
                        ImageTransformer,
                        BERTTextEncoder,
                    },
                ),
            )

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

    def _log_iteration_times(self, iteration_times):
        profile_warmup_steps = config.get("profile_warmup_steps", 100)
        start_idx = (
            profile_warmup_steps
            if profile_warmup_steps < self.config.training.max_steps
            else 0
        )
        iteration_times = iteration_times[start_idx:]
        avg_it_time = np.mean(iteration_times)
        avg_throughput = (
            config.training.batch_size * dist.get_world_size()
        ) / avg_it_time
        print0(f"Average over {len(iteration_times)} steps")
        print0(f"Average iteration time {round(avg_it_time,4)}")
        print0(f"Average throughput {round(avg_throughput,4)}")

    def train(self) -> None:
        print0(OmegaConf.to_container(self.config.training))
        self.model = self.create_model()
        model = self.model

        optimizer, scheduler = get_optimizer(
            model,
            **self.config.training.optimizer,
        )

        iteration_times = []

        while True:
            t0 = time.time()
            self.epochs += 1
            dataloader = self.datamodule.train_dataloader()
            dataloader.set_epoch(self.epochs)

            for i, data in enumerate(dataloader):
                torch.cuda.reset_peak_memory_stats()

                self.steps += 1

                if self.config.training.max_steps < self.steps:
                    if self.rank == 0:
                        self._log_iteration_times(iteration_times)
                    print0("Max steps reached, exiting")
                    return

                model.train()
                data = self.preprocess_data(data)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(
                    dtype=self.half_dtype, enabled=bool(self.scaler)
                ):
                    output = model(data)
                print0(
                    f"after forward pass {torch.cuda.memory_allocated()/1024**3:.3} GB"
                )
                self.log(
                    "stats/fwd memory alloc",
                    torch.cuda.memory_allocated() / 1024**3,
                )
                self.log(
                    "stats/fwd memory reserved",
                    torch.cuda.memory_reserved() / 1024**3,
                )

                total_loss = self.calculate_loss(output)

                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                scheduler.step()
                torch.cuda.synchronize()
                t1 = time.time()
                batch_time = t1 - t0
                batch_size = config.training.batch_size * dist.get_world_size()
                items_time = batch_size / (t1 - t0)

                t0 = t1
                self.log("stats/sec per batch", batch_time)
                self.log("stats/items per sec", items_time)

                total_loss = total_loss.detach()
                dist.reduce(total_loss, dst=0)

                if self.rank == 0:
                    norm_total_loss = total_loss.item() / dist.get_world_size()

                    print(
                        f"epoch: {self.epochs} step {self.steps} loss: {norm_total_loss:.4}"
                    )
                    self.log("train/loss", norm_total_loss)
                    self.log("stats/batch_size", batch_size)

                    iteration_times.append(batch_time)

                    cuda_info = torch.cuda.memory_stats()
                    print("cuda alloc retries ", cuda_info.get("num_alloc_retries", 0))

                self.log(
                    "stats/max_gpu_allocated_gb",
                    torch.cuda.max_memory_allocated() / 1024**3,
                )
                # TODO implement imagenet eval
                # TODO implement checkpoint saving

                self.validate()

    def validate(self):
        if self.steps % self.config.training.validation_steps != 0 or self.steps == 0:
            return

        model = self.model
        model.eval()
        print0("evaluating")

        validation_loader = self.datamodule.val_dataloader()
        validation_loss = torch.Tensor([0]).to(self.device)

        for data in validation_loader:
            data = self.preprocess_data(data)
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    dtype=self.half_dtype, enabled=bool(self.scaler)
                ):
                    output = model(data)
                    total_loss = self.calculate_loss(output, validation=True)
                    validation_loss += total_loss.detach()

        dist.reduce(validation_loss, dst=0)
        norm_validation_loss = validation_loss.item() / dist.get_world_size()

        print0(f"step {self.steps} EVAL loss: {norm_validation_loss:.4}")

    def imagenet_validate(self):
        print0("imagenet validation")
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                dtype=self.half_dtype, enabled=bool(self.scaler)
            ):
                metrics = run_imagenet_zero_shot(
                    self.model,
                    self.imagenet_val_dataloader,
                    self.device,
                    self.imagenet_val_text_transform,
                )
                if metrics is not None:
                    for key in metrics:
                        self.log(
                            f"val/imagenet/{key}",
                            metrics[key],
                            always_log=True,
                        )


if __name__ == "__main__":
    datasets.logging.set_verbosity_error()  # too spammy

    config: FLAVAArguments = build_config()
    if config.training.enable_tf32:
        enable_tf32()

    trainer = Trainer(config)
    trainer.train()
