# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from callbacks.multimodal_eval import MultimodalEvalCallback
from data import (
    ImageDataModule,
    MLMDataModule,
    MultiDataModule,
    VLDataModule,
)
from definitions import FLAVAArguments
from model import FLAVAPreTrainingLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
<<<<<<< HEAD
<<<<<<< HEAD
from utils import build_config, build_datamodule_kwargs
=======


AVAIL_GPUS = 2
SEED = -1

IMAGENET_TAR_PATH = ""
NUM_WORKERS = 4
MAX_STEPS = 450000
BATCH_SIZE = 8
ALLOW_UNEVEN_BATCHES = False
>>>>>>> 6fb41e7 ([fix,refactor] ImageNet HF support and fixes)
=======
from utils import build_config, build_datamodule_kwargs
>>>>>>> b2a1f6a ([feat] Add support for configuration system along with README)


def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    datamodules = []
    imagenet_datamodule = ImageDataModule(
<<<<<<< HEAD
<<<<<<< HEAD
        **build_datamodule_kwargs(config.datasets.image, config.training)
=======
        [
            HFDatasetInfo(
                "aps/imagenet2012", extra_kwargs={"data_dir": IMAGENET_TAR_PATH}
            )
        ],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    mlm_datamodule = MLMDataModule(
        [HFDatasetInfo("wikitext", subset="wikitext-103-raw-v1")],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
    )
    vl_datamodule = VLDataModule(
        train_dataset_infos=[
            HFDatasetInfo(
                key="red_caps",
                subset="jellyfish",
                rename_columns=[("caption", "text")],
            )
        ],
        val_dataset_infos=[
            HFDatasetInfo(
                key="red_caps",
                subset="jellyfish",
                rename_columns=[("caption", "text")],
                split_key_mapping={"validation": "train"},
            )
        ],
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        allow_unenven_batchs=ALLOW_UNEVEN_BATCHES,
>>>>>>> 6fb41e7 ([fix,refactor] ImageNet HF support and fixes)
=======
        **build_datamodule_kwargs(config.datasets.image, config.training)
>>>>>>> b2a1f6a ([feat] Add support for configuration system along with README)
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

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
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
