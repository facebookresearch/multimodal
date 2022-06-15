# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from flava.data import TextDataModule, TorchVisionDataModule
from flava.data.datamodules import VLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAClassificationLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from utils import build_config, build_datamodule_kwargs

AVAIL_GPUS = 1
SEED = -1
NUM_CLASSES = 2

NUM_WORKERS = 4
MAX_STEPS = 24000
BATCH_SIZE = 32


def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    assert len(config.datasets.selected) == 1
    if "image" in config.datasets.selected:
        datamodule = TorchVisionDataModule(
            **build_datamodule_kwargs(config.datasets.image, config.training)
        )
    elif "text":
        datamodule = TextDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training)
        )
    else:
        datamodule = VLDataModule(
            **build_datamodule_kwargs(config.datasets.vl, config.training),
            finetuning=True,
        )

    datamodule.setup("fit")

    model = FLAVAClassificationLightningModule(
        num_classes=config.datasets.num_classes,
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy="ddp",
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
