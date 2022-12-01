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
from torchmultimodal.transforms.video_transform import VideoTransform
from torchtext.models.bert.bundler import BertTextTransform


def get_yaml_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running VideoCLIP evaluation"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    return conf


def evaluate():
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
        shuffle=args.datamodule_args.shuffle,
    )

    model = VideoCLIPLightningModule.load_from_checkpoint(
        args.checkpoint_path,
        **vars(args.lightningmodule_args),
        **vars(args.videoclip_args),
    )

    trainer = Trainer(accelerator=args.accelerator, devices=1)
    trainer.test(model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    evaluate()
