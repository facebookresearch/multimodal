# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.multiprocessing as mp
import torch.distributed as dist
import os
import torch
from common.data import MultiDataModule
from flava.callbacks.multimodal_eval import MultimodalEvalCallback
from flava.data import ImageDataModule, MLMDataModule, VLDataModule
from flava.definitions import FLAVAArguments
from flava.model import (
    FLAVAPreTrainingLightningModule,
    FLAVAPreTrainingLightningModuleFSDP,
)
from flava.utils import build_config, build_datamodule_kwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.profiler import PyTorchProfiler
from torch.multiprocessing import spawn


def main(i=-1, file_str=None):
#    print(f"spawned subprocesses with local rank {i}")
#    filename = f"/tmp/{file_str}"
#    print(filename)
#    init_method = f"file://tmp{filename}"
#    torch.cuda.set_device(i)
#    os.environ["LOCAL_RANK"] = str(i)
#    os.environ["RANK"] = str(i)
#    print("before init pg")
#    dist.init_process_group(
#        backend="nccl", init_method=init_method, rank=i, world_size=8
#    )
#    print("Pg init")
#    rank = torch.distributed.get_rank()
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    datamodules = []
    #if rank != 0:
    #    torch.distributed.barrier()
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

    datamodule = MultiDataModule(datamodules)

    profiler = PyTorchProfiler()

    datamodule.setup("fit")
    #if rank == 0:
    #    dist.barrier()
    if config.training.lightning.strategy == "fsdp_native":
        print("Using FSDP")
        model = FLAVAPreTrainingLightningModuleFSDP(
            learning_rate=config.training.learning_rate,
            adam_eps=config.training.adam_eps,
            adam_weight_decay=config.training.adam_weight_decay,
            adam_betas=config.training.adam_betas,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.lightning.max_steps,
            **config.model,
        )
    else:
        print(f"Using {config.training.lightning.strategy}")
        model = FLAVAPreTrainingLightningModule(
            learning_rate=config.training.learning_rate,
            adam_eps=config.training.adam_eps,
            adam_weight_decay=config.training.adam_weight_decay,
            adam_betas=config.training.adam_betas,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.lightning.max_steps,
            **config.model,
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule),
    ]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint)
            )
        )

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        profiler=profiler,
        callbacks=callbacks,
    )
    ckpt_path = config.training.lightning_load_from_checkpoint

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
    #if "MASTER_ADDR" not in os.environ:
    #    os.environ["MASTER_ADDR"] = "localhost"
    #if "MASTER_PORT" not in os.environ:
    #    os.environ["MASTER_PORT"] = "29501"
    #assert torch.cuda.is_available()
    #nprocs = torch.cuda.device_count()
    #import string
    #import random
    #file_str = "".join(
    #    random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase)
    #    for _ in range(20)
    #)
    #mp.spawn(main, nprocs=nprocs, args=(file_str,))
