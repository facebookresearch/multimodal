# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from common.data import MultiDataModule
from flava.callbacks.multimodal_eval import MultimodalEvalCallback
from flava.data import ImageDataModule, MLMDataModule, VLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAPreTrainingLightningModule
from flava.utils import build_config, build_datamodule_kwargs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

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
        callbacks=callbacks,
    )
    ckpt_path = config.training.lightning_load_from_checkpoint

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
