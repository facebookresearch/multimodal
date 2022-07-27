# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, Callable, Dict, Optional, Tuple

from examples.mugen.data.audio_utils import AUDIO_SAMPLE_LENGTH, AUDIO_SAMPLE_RATE
from examples.mugen.data.mugen_datamodules import MUGENDataModule
from examples.mugen.data.mugen_dataset import MUGENDatasetArgs

from examples.mugen.retrieval.model import VideoCLIPLightningModule
from examples.mugen.retrieval.video_clip import PRETRAINED_S3D_KINETICS400_URL
from pytorch_lightning import Trainer

from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.video_transform import VideoTransform


arg_structure = {
    "datamodule": [
        {
            "name_or_flags": "--text_transform",
            "type": Optional[Callable],
            "default": BertTextTransform(),
        },
        {
            "name_or_flags": "--video_transform",
            "type": Optional[Callable],
            "default": VideoTransform(),
        },
        {
            "name_or_flags": "--audio_transform",
            "type": Optional[Callable],
            "default": None,
        },
        {"name_or_flags": "--batch_size", "type": int, "default": 16},
        {"name_or_flags": "--num_workers", "type": int, "default": 4},
        {"name_or_flags": "--shuffle", "type": bool, "default": False},
    ],
    "lightningmodule": [
        {"name_or_flags": "--logit_scale", "type": float, "default": 0.07},
        {"name_or_flags": "--logit_scale_max", "type": float, "default": 100.0},
        {"name_or_flags": "--recall_ks", "type": Tuple[int], "default": (1, 5, 10)},
    ],
    "dataset": [
        {
            "name_or_flags": "--data_path",
            "type": str,
            "default": "datasets/coinrun/coinrun_dataset_jsons/release",
        },
        {"name_or_flags": "--sample_every_n_frames", "type": int, "default": 3},
        {"name_or_flags": "--sequence_length", "type": int, "default": 32},
        {"name_or_flags": "--resolution", "type": int, "default": 256},
        {
            "name_or_flags": "--audio_sample_rate",
            "type": int,
            "default": AUDIO_SAMPLE_RATE,
        },
        {
            "name_or_flags": "--audio_sample_length",
            "type": int,
            "default": AUDIO_SAMPLE_LENGTH,
        },
        {"name_or_flags": "--bbox_smap_for_agent", "type": bool, "default": False},
        {"name_or_flags": "--bbox_smap_for_monsters", "type": bool, "default": False},
        {"name_or_flags": "--use_manual_annotation", "type": bool, "default": False},
        {"name_or_flags": "--use_auto_annotation", "type": bool, "default": False},
        {"name_or_flags": "--use_downsampled_trainset", "type": bool, "default": False},
        {"name_or_flags": "--fixed_start_idx", "type": bool, "default": False},
        {"name_or_flags": "--get_game_frame", "type": bool, "default": True},
        {"name_or_flags": "--get_seg_map", "type": bool, "default": False},
        {"name_or_flags": "--get_text_desc", "type": bool, "default": True},
        {"name_or_flags": "--get_audio", "type": bool, "default": False},
        {"name_or_flags": "--debug", "type": bool, "default": False},
    ],
    "videoclip": [
        {"name_or_flags": "--text_pretrained", "type": bool, "default": False},
        {"name_or_flags": "--text_trainable", "type": bool, "default": False},
        {
            "name_or_flags": "--text_model_name",
            "type": str,
            "default": "distilbert-base-uncased",
        },
        {
            "name_or_flags": "--text_model_config",
            "type": Optional[Dict[str, Any]],
            "default": None,
        },
        {"name_or_flags": "--text_padding_value", "type": int, "default": 0},
        {"name_or_flags": "--video_pretrained", "type": bool, "default": False},
        {"name_or_flags": "--video_trainable", "type": bool, "default": False},
        {
            "name_or_flags": "--video_pretrain_path",
            "type": str,
            "default": PRETRAINED_S3D_KINETICS400_URL,
        },
        {"name_or_flags": "--proj_out_dim", "type": int, "default": 256},
        {"name_or_flags": "--proj_dropout", "type": float, "default": 0.1},
    ],
    "evaluation": [
        {"name_or_flags": "--accelerator", "type": str, "default": "auto"},
        {"name_or_flags": "--devices", "type": int, "default": 1},
        {
            "name_or_flags": "--checkpoint_path",
            "type": str,
            "default": "lightning_videoclip_mugen_ckpt.pt",
        },
    ],
}


def parse_all_args():
    parser = argparse.ArgumentParser()
    for argument_dicts in arg_structure.values():
        for argument in argument_dicts:
            # Can't unpack `**argument` directly because `add_argument` requires a positional argument
            parser.add_argument(
                argument["name_or_flags"],
                type=argument["type"],
                default=argument["default"],
            )
    args = parser.parse_args()
    return args


def get_args_from_group(args, group_name):
    """Utility for grouping arguments based on the structure in ``arg_structure``"""

    def attr_name(arg_name):
        """Returns the name of an argument without the ``--`` prefix"""
        return arg_name[2:]

    return argparse.Namespace(
        **{
            attr_name(arg["name_or_flags"]): getattr(
                args, attr_name(arg["name_or_flags"])
            )
            for arg in arg_structure[group_name]
        }
    )


def evaluate():
    args = parse_all_args()
    datamodule_args = get_args_from_group(args, "datamodule")
    lightningmodule_args = get_args_from_group(args, "lightningmodule")
    dataset_args = get_args_from_group(args, "dataset")
    videoclip_args = get_args_from_group(args, "videoclip")
    evaluation_args = get_args_from_group(args, "evaluation")

    dataset_args = MUGENDatasetArgs(**vars(dataset_args))
    datamodule = MUGENDataModule(dataset_args, **vars(datamodule_args))

    model = VideoCLIPLightningModule(
        **vars(lightningmodule_args), **vars(videoclip_args)
    )
    model = model.load_from_checkpoint(evaluation_args.checkpoint_path)

    trainer = Trainer(
        accelerator=evaluation_args.accelerator, devices=evaluation_args.devices
    )
    trainer.test(model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    evaluate()
