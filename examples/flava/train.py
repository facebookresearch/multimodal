# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from common.dataset_utils import MultiDataModule
from common.dataset_utils.iteration_strategies import iteration_strategy_factory
from flava.callbacks.multimodal_eval import MultimodalEvalCallback
from flava.data import ImageDataModule, MLMDataModule, VLDataModule
from flava.definitions import FLAVAArguments
from flava.model import FLAVAPreTrainingLightningModule
from flava.utils import build_config, build_datamodule_kwargs

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor


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

    datamodule = MultiDataModule(
        datamodules, iteration_strategy_factory(config.datasets.iteration_strategy)
    )

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
