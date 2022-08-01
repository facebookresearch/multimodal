# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

from examples.mugen.data.audio_utils import AUDIO_SAMPLE_LENGTH, AUDIO_SAMPLE_RATE

from examples.mugen.data.mugen_datamodules import MUGENDataModule
from examples.mugen.data.mugen_dataset import MUGENDatasetArgs

from examples.mugen.retrieval.model import VideoCLIPLightningModule
from examples.mugen.retrieval.video_clip import PRETRAINED_S3D_KINETICS400_URL
from pytorch_lightning import Trainer

from torchmultimodal.transforms.bert_text_transform import BertTextTransform
from torchmultimodal.transforms.video_transform import VideoTransform


arg_structure = {
    "datamodule": {
        "batch_size": {"type": int, "default": 16},
        "num_workers": {"type": int, "default": 4},
        "shuffle": {"action": "store_true"},
    },
    "lightningmodule": {
        "logit_scale": {"type": float, "default": 0.07},
        "logit_scale_max": {"type": float, "default": 100.0},
    },
    "dataset": {
        "data_path": {
            "type": str,
            "default": "datasets/coinrun/coinrun_dataset_jsons/release",
        },
        "sample_every_n_frames": {"type": int, "default": 3},
        "sequence_length": {"type": int, "default": 32},
        "resolution": {"type": int, "default": 224},
        "audio_sample_rate": {"type": int, "default": AUDIO_SAMPLE_RATE},
        "audio_sample_length": {"type": int, "default": AUDIO_SAMPLE_LENGTH},
        "bbox_smap_for_agent": {"action": "store_true"},
        "bbox_smap_for_monsters": {"action": "store_true"},
        "use_manual_annotation": {"action": "store_true"},
        "use_auto_annotation": {"action": "store_true"},
        "use_downsampled_trainset": {"action": "store_true"},
        "fixed_start_idx": {"action": "store_true"},
        "get_game_frame": {"action": "store_true"},
        "get_seg_map": {"action": "store_true"},
        "get_text_desc": {"action": "store_true"},
        "get_audio": {"action": "store_true"},
        "debug": {"action": "store_true"},
    },
    "videoclip": {
        "text_pretrained": {"action": "store_true"},
        "text_trainable": {"action": "store_true"},
        "text_model_name": {"type": str, "default": "distilbert-base-uncased"},
        "text_model_config": {"type": json.loads, "default": None},
        "text_padding_value": {"type": int, "default": 0},
        "video_pretrained": {"action": "store_true"},
        "video_trainable": {"action": "store_true"},
        "video_pretrain_path": {"type": str, "default": PRETRAINED_S3D_KINETICS400_URL},
        "proj_out_dim": {"type": int, "default": 256},
        "proj_dropout": {"type": float, "default": 0.1},
    },
    "evaluation": {
        "checkpoint_path": {
            "type": str,
            "default": "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/videoclip_lightning_mugen.pt",
        },
    },
}


def parse_all_args():
    parser = argparse.ArgumentParser()
    for arguments_dict in arg_structure.values():
        for arg_name, arg_options in arguments_dict.items():
            parser.add_argument("--" + arg_name, **arg_options)
    args = parser.parse_args()
    return args


def get_args_from_group(args, group_name):
    """Utility for grouping arguments based on the structure in ``arg_structure``"""
    return argparse.Namespace(
        **{
            arg_name: getattr(args, arg_name)
            for arg_name in arg_structure[group_name].keys()
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
    datamodule = MUGENDataModule(
        dataset_args,
        text_transform=BertTextTransform(),
        video_transform=VideoTransform(),
        **vars(datamodule_args),
    )

    model = VideoCLIPLightningModule(
        **vars(lightningmodule_args), **vars(videoclip_args)
    )
    model = model.load_from_checkpoint(evaluation_args.checkpoint_path)

    trainer = Trainer(accelerator="auto", devices=1)
    trainer.test(model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    evaluate()
