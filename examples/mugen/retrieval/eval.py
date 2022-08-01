# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from examples.mugen.data.mugen_datamodules import MUGENDataModule
from examples.mugen.data.mugen_dataset import MUGENDatasetArgs

from examples.mugen.retrieval.model import VideoCLIPLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.video_transform import VideoTransform


def get_yaml_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running VideoCLIP evaluation"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.to_container(yaml_conf)


def evaluate():
    args = get_yaml_config()
    datamodule_args = args["datamodule_args"]
    lightningmodule_args = args["lightningmodule_args"]
    dataset_args = args["dataset_args"]
    videoclip_args = args["videoclip_args"]
    evaluation_args = args["evaluation_args"]

    dataset_args = MUGENDatasetArgs(get_audio=False, **dataset_args)
    datamodule = MUGENDataModule(
        dataset_args,
        text_transform=BertTextTransform(**datamodule_args["bert_text_transform"]),
        video_transform=VideoTransform(**datamodule_args["video_transform"]),
        batch_size=datamodule_args["batch_size"],
        num_workers=datamodule_args["num_workers"],
        shuffle=datamodule_args["shuffle"],
    )

    model = VideoCLIPLightningModule(**lightningmodule_args, **videoclip_args)
    model = model.load_from_checkpoint(evaluation_args["checkpoint_path"])

    trainer = Trainer(
        accelerator=evaluation_args["accelerator"], devices=evaluation_args["devices"]
    )
    trainer.test(model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    evaluate()
