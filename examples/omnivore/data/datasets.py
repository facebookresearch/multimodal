# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import PIL
import scipy.io
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset


class OmnivoreKinetics(torchvision.datasets.kinetics.Kinetics):
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label, video_idx


class OmnivoreSunRgbdDatasets(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, split="train"):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUNRGBD"
        self._meta_dir = Path(self.root) / "SUNRGBDtoolbox"

        if not self._check_exists():
            print(f"data_dir: {self._data_dir}\nmeta_dir: {self._meta_dir}")
            raise RuntimeError("Dataset not found.")

        # Get the param from https://github.com/facebookresearch/omnivore/issues/12
        self.sensor_to_params = {
            "kv1": {
                "baseline": 0.075,
            },
            "kv1_b": {
                "baseline": 0.075,
            },
            "kv2": {
                "baseline": 0.075,
            },
            "realsense": {
                "baseline": 0.095,
            },
            "xtion": {
                "baseline": 0.095,  # guessed based on length of 18cm for ASUS xtion v1
            },
        }
        # Omnivore only use these selected 19 classes
        self.classes = [
            "bathroom",
            "bedroom",
            "classroom",
            "computer_room",
            "conference_room",
            "corridor",
            "dining_area",
            "dining_room",
            "discussion_area",
            "furniture_store",
            "home_office",
            "kitchen",
            "lab",
            "lecture_theatre",
            "library",
            "living_room",
            "office",
            "rest_space",
            "study_space",
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        allsplit_filepath = self._meta_dir / "traintestSUNRGBD/allsplit.mat"
        allsplit_mat = scipy.io.loadmat(allsplit_filepath)

        # The original filepath on the "allsplit.mat" has the prefix from author machine that need to be replaced
        ori_prefix = "/n/fs/sun3d/data/SUNRGBD/"
        if split == "train":
            self.image_dirs = [
                self._data_dir / x[0][len(ori_prefix) :]
                for x in allsplit_mat["alltrain"][0]
            ]
        elif split == "val":
            self.image_dirs = [
                self._data_dir / x[0][len(ori_prefix) :]
                for x in allsplit_mat["alltest"][0]
            ]

        # Filter to use only chosen 19 classes
        self.image_dirs = [
            x
            for x in self.image_dirs
            if self._get_sunrgbd_scene_class(x) in self.class_to_idx
        ]

    def _check_exists(self):
        return self._data_dir.is_dir() and self._meta_dir.is_dir()

    def __len__(self):
        return len(self.image_dirs)

    def _get_disparity_tensor(self, image_dir):
        # Using depth_bfx, but maybe can also consider just using depth
        image_dir = Path(image_dir)
        depth_dir = image_dir / "depth_bfx"
        intrinsics_file = image_dir / "intrinsics.txt"
        depth_path = depth_dir / os.listdir(depth_dir)[0]

        sensor_type = image_dir.relative_to(self._data_dir).parts[0]
        baseline = self.sensor_to_params[sensor_type]["baseline"]
        with open(intrinsics_file, "r") as fin:
            lines = fin.readlines()
            focal_length = float(lines[0].strip().split()[0])

        img_depth = PIL.Image.open(depth_path)
        tensor_depth = T.ToTensor()(img_depth)
        tensor_disparity = baseline * focal_length / (tensor_depth / 1000.0)
        return tensor_disparity

    def _read_sunrgbd_image(self, image_dir):
        rgb_dir = os.path.join(image_dir, "image")
        rgb_path = os.path.join(rgb_dir, os.listdir(rgb_dir)[0])
        img_rgb = PIL.Image.open(rgb_path)
        tensor_rgb = T.ToTensor()(img_rgb)

        tensor_d = self._get_disparity_tensor(image_dir)

        tensor_rgbd = torch.cat((tensor_rgb, tensor_d), dim=0)
        return tensor_rgbd

    def _get_sunrgbd_scene_class(self, image_dir):
        with open(os.path.join(image_dir, "scene.txt"), "r") as fin:
            scene_class = fin.read().strip()
        return scene_class

    def __getitem__(self, idx):
        # return tuple of image (H W C==4) and scene class index
        image_dir = self.image_dirs[idx]
        x_rgbd = self._read_sunrgbd_image(image_dir)
        scene_class = self._get_sunrgbd_scene_class(image_dir)
        scene_idx = self.class_to_idx[scene_class]

        if self.transform:
            x_rgbd = self.transform(x_rgbd)

        if self.target_transform:
            scene_idx = self.target_transform(scene_idx)

        return x_rgbd, scene_idx


class ConcatIterable:
    """
    ConcatIterable is used to group iterable object.
    When user iterate on this object, we will sample random iterable and return their
    item with coresponding output_key.
    With repeat_factors, user can do upsampling or downsampling to the iterables.
    We mainly use this class to concat different data loader during training.

    Args:
        iterables: the iterable objects that will be grouped
        output_keys: List of keys that is used to identify the iterable output.
            The list length should be the same as number of iterables.
        repeat_factors: List of numbers that represent the upsampling / downsampling factor
            to the coresponding iterables. Should have same length as iterables.
        seed: the seed for randomness
    """

    def __init__(self, iterables, output_keys, repeat_factors, seed=42):
        self.iterables = iterables
        self.output_keys = output_keys
        self.repeat_factors = repeat_factors
        self.seed = seed
        self.num_iterables = len(self.iterables)
        assert self.num_iterables == len(output_keys)
        assert self.num_iterables == len(repeat_factors)

        # The iterator len is adjusted with repeat_factors
        self.iterator_lens = [
            int(repeat_factors[i] * len(itb)) for i, itb in enumerate(self.iterables)
        ]
        self.max_total_steps = sum(self.iterator_lens)
        self.indices = None
        self.iterators = None

        # self.step_counter == None indicate that self.indices are not yet initialized
        self.step_counter = None

    def init_indices(self, epoch=0, shuffle=False):
        # We should initiate indices for each epoch, especially if we want to shuffle
        self.step_counter = 0

        self.iterators = [iter(dl) for dl in self.iterables]
        self.indices = torch.cat(
            [
                torch.ones(self.iterator_lens[i], dtype=torch.int32) * i
                for i in range(self.num_iterables)
            ]
        )
        assert self.max_total_steps == len(self.indices)

        if shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            shuffle_indices = torch.randperm(len(self.indices), generator=g)
            self.indices = self.indices[shuffle_indices]

    def __next__(self):
        if self.step_counter is None:
            # Initiate the indices without shuffle as default!
            self.init_indices()
        if self.step_counter >= self.max_total_steps:
            raise StopIteration

        idx = self.indices[self.step_counter]
        output_key = self.output_keys[idx]
        # print(idx)
        try:
            batch = next(self.iterators[idx])
        except StopIteration:
            # We cycle over the data_loader to the beginning. This can happen when repeat_factor > 1
            # Take note that in this case we always use same shuffling from same data_loader in an epoch
            self.iterators[idx] = iter(self.iterables[idx])
            batch = next(self.iterators[idx])

        self.step_counter += 1
        # Return batch and output_key
        return batch, output_key

    def __len__(self):
        return self.max_total_steps

    def __iter__(self):
        return self
