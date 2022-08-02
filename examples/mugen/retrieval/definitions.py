# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from examples.mugen.data.audio_utils import AUDIO_SAMPLE_LENGTH, AUDIO_SAMPLE_RATE

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
    mean: Tuple[float] = tuple(DEFAULT_MEAN)
    std: Tuple[float] = tuple(DEFAULT_STD)
    resize_shape: Tuple[int, int] = DEFAULT_RESIZE_SHAPE


@dataclass
class MUGENDatasetArgs:
    data_path: str = "datasets/coinrun/coinrun_dataset_jsons/release"
    asset_path: str = "datasets/coinrun/assets"
    sample_every_n_frames: int = 3
    sequence_length: int = 32
    resolution: int = 256
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    audio_sample_length: int = AUDIO_SAMPLE_LENGTH
    bbox_smap_for_agent: bool = (
        True  # render smap for mugen (and shield) as bounding boxes
    )
    bbox_smap_for_monsters: bool = True  # render smap for monsters as bounding boxes
    use_manual_annotation: bool = False  # if True will only use videos with manual annotation and skip those without
    use_auto_annotation: bool = (
        True  # if True will only use videos with auto annotation and skip those without
    )
    use_downsampled_trainset: bool = (
        False  # if True will only use downsampled training set
    )
    fixed_start_idx: bool = True  # fx starting game frame idx to 0
    get_game_frame: bool = False  # load video data
    get_seg_map: bool = False  # load semantic map
    get_text_desc: bool = False  # load text data
    get_audio: bool = (
        False  # load full mix audio for each video, for audio generation models
    )
    debug: bool = False


@dataclass
class DataModuleArgs:
    batch_size: int = 16
    num_workers: int = 4
    shuffle: bool = False
    bert_text_transform: BertTextTransformArgs = BertTextTransformArgs()
    video_transform: VideoTransformArgs = VideoTransformArgs()


@dataclass
class LightningModuleArgs:
    logit_scale: float = 0.07
    logit_scale_max: float = 100.0


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
        resolution=224,
        fixed_start_idx=False,
        use_manual_annotation=True,
        use_auto_annotation=False,
    )
    datamodule_args: DataModuleArgs = DataModuleArgs()
    lightningmodule_args: LightningModuleArgs = LightningModuleArgs()
    videoclip_args: VideoCLIPArgs = VideoCLIPArgs()
    checkpoint_path: str = "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/videoclip_lightning_mugen.pt"
