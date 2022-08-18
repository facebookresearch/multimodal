# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from examples.mugen.data.mugen_datamodules import MUGENDataModule
from examples.mugen.data.mugen_dataset import MUGENDatasetArgs
from examples.mugen.retrieval.model import VideoCLIPLightningModule
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.video_transform import VideoTransform


def get_yaml_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running VideoCLIP training"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    return conf


def train():
    args = get_yaml_config()

    dataset_args: MUGENDatasetArgs = args.dataset_args
    datamodule = MUGENDataModule(
        dataset_args,
        text_transform=BertTextTransform(
            **vars(args.datamodule_args.bert_text_transform)
        ),
        video_transform=VideoTransform(**vars(args.datamodule_args.video_transform)),
        batch_size=args.datamodule_args.batch_size,
        num_workers=args.datamodule_args.num_workers,
    )

    model = VideoCLIPLightningModule(
        **vars(args.lightningmodule_args),
        **vars(args.videoclip_args),
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=-1)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy="ddp_find_unused_parameters_false",
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(args.shuffle_train),
        val_dataloaders=datamodule.val_dataloader(args.shuffle_test),
    )


if __name__ == "__main__":
    train()
