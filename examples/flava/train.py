# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from data import (
    ImageDataModule,
    MLMDataModule,
    HFDatasetInfo,
    VLDataModule,
    MultiDataModule,
)
from examples.flava.callbacks.multimodal_eval import MultimodalEvalCallback
from model import FLAVALightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from examples.flava.callbacks.multimodal_eval import MultimodalEvalCallback


AVAIL_GPUS = 2
SEED = -1

IMAGENET_TRAIN_ROOT = ""
IMAGENET_VAL_ROOT = ""
NUM_WORKERS = 4
MAX_STEPS = 450000
BATCH_SIZE = 8
ALLOW_UNEVEN_BATCHES = False


def main():
    if SEED != -1:
        seed_everything(SEED, workers=True)

    imagenet_datamodule = ImageDataModule(
        train_root=IMAGENET_TRAIN_ROOT,
        val_root=IMAGENET_VAL_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    mlm_datamodule = MLMDataModule(
        [HFDatasetInfo("wikitext", "wikitext-103-raw-v1")],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    vl_datamodule = MultiDataModule(
        [
            VLDataModule(
                train_dataset_infos=[
                    HFDatasetInfo(
                        key="red_caps",
                        subset="mycology",
                        rename_columns=[("caption", "text")],
                    )
                ],
                val_dataset_infos=[
                    HFDatasetInfo(
                        key="red_caps",
                        subset="mycology",
                        rename_columns=[("caption", "text")],
                        split_key_mapping={"validation": "train"},
                    )
                ],
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
            )
        ]
    )
    datamodule = MultiDataModule([imagenet_datamodule, mlm_datamodule, vl_datamodule])

    datamodule.setup("fit")
    model = FLAVALightningModule()

    trainer = Trainer(
        max_steps=MAX_STEPS,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=50,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule),
        ],
        strategy="ddp",
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
