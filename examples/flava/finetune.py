# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchvision
from data import (
    TorchVisionDataModule,
    TorchVisionDatasetInfo,
    HFDatasetInfo,
    TextDataModule,
)
from model import FLAVAClassificationLightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

AVAIL_GPUS = 1
SEED = -1
NUM_CLASSES = 2

NUM_WORKERS = 4
MAX_STEPS = 24000
BATCH_SIZE = 32


rendered_sst2_info = TorchVisionDatasetInfo(
    key="rendered_sst2",
    class_ptr=torchvision.datasets.RenderedSST2,
)


def main():
    if SEED != -1:
        seed_everything(SEED, workers=True)

    datamodule = TorchVisionDataModule(
        rendered_sst2_info,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    datamodule.setup("fit")
    model = FLAVAClassificationLightningModule(num_classes=NUM_CLASSES)

    trainer = Trainer(
        max_steps=MAX_STEPS,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=50,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy="ddp",
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
