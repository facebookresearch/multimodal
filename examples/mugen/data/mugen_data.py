# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import NamedTuple

import numpy as np
import torch
import torch.utils.data as data
from data.coinrun.construct_from_json import (
    define_semantic_color_map,
    draw_game_frame,
    generate_asset_paths,
    load_assets,
    load_bg_asset,
)

from data.coinrun.game import Game

from .audio_utils import AUDIO_SAMPLE_LENGTH, AUDIO_SAMPLE_RATE, load_audio


class MUGENDatasetArgs(NamedTuple):
    data_path: str = "datasets/coinrun/coinrun_dataset_jsons/release"
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
    get_game_frame: bool = True
    get_seg_map: bool = True
    get_text_desc: bool = True
    get_audio: bool = (
        True  # load full mix audio for each video, for audio generation models
    )
    debug: bool = False


class MUGENDataset(data.Dataset):
    """Dataset class to interface the MUGEN dataset.

    Args:
        split (str): dataset split. Defines the json file from which to read metadata.
            E.g. ``"train"``, ``"valid"``, or ``"test"``.
        args (MUGENDatasetArgs): other arguments related to loading data from files.
    """

    def __init__(
        self,
        split: str,
        args: MUGENDatasetArgs,
    ):
        super().__init__()
        self.args = args
        self.train = split == "train"
        self.max_label = 21

        assert (
            self.args.get_game_frame or self.args.get_audio or self.args.get_text_desc
        ), "Need to return at least one of game frame, audio, or text desc"
        if args.use_downsampled_trainset and split == "train":
            dataset_json_file = os.path.join(
                self.args.data_path, f"{split}_downsampled.json"
            )
        else:
            dataset_json_file = os.path.join(self.args.data_path, f"{split}.json")
        print(f"LOADING FROM JSON FROM {dataset_json_file}...")
        with open(dataset_json_file, "r") as f:
            all_data = json.load(f)
        if args.debug:
            all_data["data"] = all_data["data"][:16]

        self.dataset_metadata = all_data["metadata"]
        self.data = []
        for data_sample in all_data["data"]:
            if (
                data_sample["video"]["num_frames"]
                > (args.sequence_length - 1) * args.sample_every_n_frames
            ):
                self.data.append(data_sample)
        print(f"NUMBER OF FILES LOADED: {len(self.data)}")

        self.init_game_assets()

    # initialize game assets
    def init_game_assets(self):
        self.game = Game()
        self.game.load_json(
            os.path.join(
                self.dataset_metadata["data_folder"], self.data[0]["video"]["json_file"]
            )
        )
        # NOTE: only supports rendering square-size coinrun frame for now
        self.game.video_res = self.args.resolution

        semantic_color_map = define_semantic_color_map(self.max_label)

        # grid size for Mugen/monsters/ground
        self.kx: float = self.game.zoom * self.game.video_res / self.game.maze_w
        self.ky: float = self.kx

        # grid size for background
        zx = self.game.video_res * self.game.zoom
        zy = zx

        # NOTE: This is a hacky solution to switch between theme assets
        # Sightly inefficient due to Mugen/monsters being loaded twice
        # but that only a minor delay during init
        # This should be revisited in future when we have more background/ground themes
        self.total_world_themes = len(self.game.background_themes)
        self.asset_map = {}
        for world_theme_n in range(self.total_world_themes):
            # reset the paths for background and ground assets based on theme
            self.game.world_theme_n = world_theme_n
            asset_files = generate_asset_paths(self.game)

            # TODO: is it worth to load assets separately for game frame and label?
            # this way game frame will has smoother character boundary
            self.asset_map[world_theme_n] = load_assets(
                asset_files, semantic_color_map, self.kx, self.ky, gen_original=False
            )

            # background asset is loaded separately due to not following the grid
            self.asset_map[world_theme_n]["background"] = load_bg_asset(
                asset_files, semantic_color_map, zx, zy
            )

    def __len__(self):
        return len(self.data)

    def get_start_end_idx(self, valid_frames=None):
        start_idx = 0
        end_idx = len(self.game.frames)
        if self.args.sequence_length is not None:
            assert (
                self.args.sequence_length - 1
            ) * self.args.sample_every_n_frames < end_idx, (
                f"not enough frames to sample {self.args.sequence_length} frames "
                + "at every {self.args.sample_every_n_frames} frame"
            )

            if self.args.fixed_start_idx:
                start_idx = 0
            else:
                if valid_frames:
                    # we are sampling frames from a full json and we need to ensure that the desired
                    # class is in the frame range we sample. Resample until this is true
                    resample = True
                    while resample:
                        start_idx = torch.randint(
                            low=0,
                            high=end_idx
                            - (self.args.sequence_length - 1)
                            * self.args.sample_every_n_frames,
                            size=(1,),
                        ).item()
                        for valid_frame_range in valid_frames:
                            if isinstance(valid_frame_range, list):
                                # character ranges
                                st_valid, end_valid = valid_frame_range
                            else:
                                # game event has a single timestamp
                                st_valid, end_valid = (
                                    valid_frame_range,
                                    valid_frame_range,
                                )
                            if (
                                end_valid >= start_idx
                                and start_idx
                                + self.args.sequence_length
                                * self.args.sample_every_n_frames
                                >= st_valid
                            ):
                                # desired class is in the sampled frame range, so stop sampling
                                resample = False
                else:
                    start_idx = torch.randint(
                        low=0,
                        high=end_idx
                        - (self.args.sequence_length - 1)
                        * self.args.sample_every_n_frames,
                        size=(1,),
                    ).item()
            end_idx = (
                start_idx + self.args.sequence_length * self.args.sample_every_n_frames
            )
        return start_idx, end_idx

    def get_game_video(self, start_idx, end_idx, alien_name="Mugen"):
        frames = []
        for i in range(start_idx, end_idx, self.args.sample_every_n_frames):
            img = draw_game_frame(
                self.game,
                i,
                self.asset_map[self.game.world_theme_n],
                self.kx,
                self.ky,
                gen_original=True,
                alien_name=alien_name,
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        return torch.vstack(frames)

    def get_game_audio(self, wav_filename):
        data, _ = load_audio(
            wav_filename,
            sr=self.args.audio_sample_rate,
            offset=0,
            duration=self.args.audio_sample_length,
        )
        data = torch.as_tensor(data).permute(1, 0)
        return data

    def get_smap_video(self, start_idx, end_idx, alien_name="Mugen"):
        frames = []
        for i in range(start_idx, end_idx, self.args.sample_every_n_frames):
            img = draw_game_frame(
                self.game,
                i,
                self.asset_map[self.game.world_theme_n],
                self.kx,
                self.ky,
                gen_original=False,
                bbox_smap_for_agent=self.args.bbox_smap_for_agent,
                bbox_smap_for_monsters=self.args.bbox_smap_for_monsters,
                alien_name=alien_name,
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 1 (sequence_length=16, resolution=256)
        return torch.unsqueeze(torch.vstack(frames), dim=3)

    def load_json_file(self, idx):
        self.game.load_json(
            os.path.join(
                self.dataset_metadata["data_folder"],
                self.data[idx]["video"]["json_file"],
            )
        )
        self.game.video_res = self.args.resolution

    def __getitem__(self, idx):
        self.load_json_file(idx)
        start_idx, end_idx = self.get_start_end_idx()
        alien_name = "Mugen"

        result_dict = {}

        if self.args.get_audio:
            wav_file = os.path.join(
                self.dataset_metadata["data_folder"],
                self.data[idx]["video"]["video_file"],
            )
            result_dict["audio"] = self.get_game_audio(wav_file)

        if self.args.get_game_frame:
            game_video = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
            result_dict["video"] = game_video

        if self.args.get_seg_map:
            seg_map_video = self.get_smap_video(
                start_idx, end_idx, alien_name=alien_name
            )
            result_dict["video_smap"] = seg_map_video

        if self.args.get_text_desc:
            # text description will be generated in the range of start and end frames
            # this means we can use full json and auto-text to train transformer too

            assert self.args.use_auto_annotation or self.args.use_manual_annotation
            if self.args.use_manual_annotation and not self.args.use_auto_annotation:
                assert (
                    len(self.data[idx]["annotations"]) > 1
                ), "need at least one manual annotation if using only manual annotations"
                # exclude the auto-text, which is always index 0
                rand_idx = (
                    torch.randint(
                        low=1, high=len(self.data[idx]["annotations"]), size=(1,)
                    ).item()
                    if self.train
                    else 1
                )
            elif not self.args.use_manual_annotation and self.args.use_auto_annotation:
                rand_idx = 0
            else:
                rand_idx = torch.randint(
                    low=0, high=len(self.data[idx]["annotations"]), size=(1,)
                ).item()

            if self.args.use_manual_annotation and not self.args.use_auto_annotation:
                assert (
                    self.data[idx]["annotations"][rand_idx]["type"] == "manual"
                ), "Should only be sampling manual annotations"

            text_desc = self.data[idx]["annotations"][rand_idx]["text"]

            result_dict["text"] = text_desc

        return result_dict
