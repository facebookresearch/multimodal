# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from examples.mugen.data.mugen_dataset import MUGENDatasetArgs

from torchmultimodal.transforms.video_transform import (
    DEFAULT_MEAN,
    DEFAULT_RESIZE_SHAPE,
    DEFAULT_STD,
    MUGEN_DEFAULT_TIME_SAMPLES,
)


@dataclass
class BertTextTransformArgs:
    vocab_file: str = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    do_lower_case: bool = True
    start_token: int = 101
    end_token: int = 102
    padding_value: int = 0


@dataclass
class VideoTransformArgs:
    time_samples: int = MUGEN_DEFAULT_TIME_SAMPLES
    mean: Tuple[float] = DEFAULT_MEAN
    std: Tuple[float] = DEFAULT_STD
    resize_shape: Tuple[int, int] = DEFAULT_RESIZE_SHAPE


@dataclass
class DataModuleArgs:
    batch_size: int = 16
    num_workers: int = 4
    bert_text_transform: BertTextTransformArgs = BertTextTransformArgs()
    video_transform: VideoTransformArgs = VideoTransformArgs()


@dataclass
class LightningModuleArgs:
    logit_scale: float = 0.07
    logit_scale_max: float = 100.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    recall_ks: Tuple[int] = (1, 5, 10)


@dataclass
class VideoCLIPArgs:
    text_pretrained: bool = False
    text_trainable: bool = False
    text_model_name: str = "distilbert-base-uncased"
    text_model_config: Optional[Dict[str, Any]] = None
    text_padding_value: int = 0
    video_pretrained: bool = False
    video_trainable: bool = False
    video_pretrain_path: str = (
        "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/S3D_kinetics400.pt"
    )
    proj_out_dim: int = 256
    proj_dropout: float = 0.1


@dataclass
class EvaluationArgs:
    dataset_args: MUGENDatasetArgs = MUGENDatasetArgs(
        get_game_frame=True,
        get_text_desc=True,
        resolution=256,
        fixed_start_idx=False,
        use_manual_annotation=True,
        use_auto_annotation=False,
    )
    datamodule_args: DataModuleArgs = DataModuleArgs()
    lightningmodule_args: LightningModuleArgs = LightningModuleArgs()
    videoclip_args: VideoCLIPArgs = VideoCLIPArgs()
    shuffle_test: bool = False
    checkpoint_path: str = "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/videoclip_lightning_mugen.pt"
    accelerator: str = "auto"


@dataclass
class TrainingArgs:
    dataset_args: MUGENDatasetArgs = MUGENDatasetArgs(
        get_game_frame=True,
        get_text_desc=True,
        resolution=224,
        fixed_start_idx=False,
        use_manual_annotation=True,
        use_auto_annotation=False,
    )
    datamodule_args: DataModuleArgs = DataModuleArgs()
    lightningmodule_args: LightningModuleArgs = LightningModuleArgs()
    videoclip_args: VideoCLIPArgs = VideoCLIPArgs()
    shuffle_train: bool = True
    shuffle_val: bool = False
    accelerator: str = "auto"
    devices: int = 4
    max_epochs: int = 1000
    log_every_n_steps: int = 100
    default_root_dir: Optional[str] = None
