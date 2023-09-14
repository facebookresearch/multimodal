# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest
import torch
from PIL import Image
from tests.test_utils import assert_expected, get_asset_path, set_rng_seed
from torchmultimodal.transforms.mae_transform import (
    AudioEvalTransform,
    AudioFineTuneTransform,
    AudioPretrainTransform,
    ImageEvalTransform,
    ImagePretrainTransform,
    MixUpCutMix,
    RandAug,
)
from torchvision import transforms

IMAGE_PATH = "tests/assets/test_image.jpg"
WAV_PATH = "tests/assets/kaldi_file_8000.wav"
MIXUP_WAV_PATH = "tests/assets/sinewave.wav"


@pytest.fixture
def image():
    return Image.open(get_asset_path("test_image.jpg"))


class TestImageEvalTransform:
    @pytest.fixture
    def transform(self):
        return ImageEvalTransform(input_size=224)

    def test_transform(self, transform, image):
        actual = transform(image)
        assert_expected(actual.size(), (3, 224, 224))
        assert_expected(actual.mean().item(), -0.4967, atol=0.0001, rtol=0.0)

    def test_transform_list(self, transform, image):
        actual = transform([image])
        assert_expected(actual.size(), (1, 3, 224, 224))
        assert_expected(actual.mean().item(), -0.4967, atol=0.0001, rtol=0.0)


class TestImagePretrainTransform:
    @pytest.fixture
    def transform(self):
        return ImagePretrainTransform(input_size=224)

    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(0)

    def test_transform(self, transform, image):
        actual = transform(image)
        assert_expected(actual.size(), (3, 224, 224))
        assert_expected(actual.mean().item(), -0.4625, atol=0.0001, rtol=0.0)

    def test_transform_list(self, transform, image):
        actual = transform([image])
        assert_expected(actual.size(), (1, 3, 224, 224))
        assert_expected(actual.mean().item(), -0.4625, atol=0.0001, rtol=0.0)


class TestMixup:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)

    @pytest.fixture
    def inputs(self):
        return torch.Tensor(
            [
                [
                    [
                        [1.0, 2.0, 1.0, 2.0],
                        [2.0, 1.0, 1.0, 2.0],
                        [2.0, 2.0, 1.0, 0.0],
                        [7.0, 5.0, 4.0, 9.0],
                    ]
                ],
                [
                    [
                        [2.0, 5.0, 1.0, 2.0],
                        [1.0, 5.0, 1.0, 2.0],
                        [3.0, 1.0, 9.0, 2.0],
                        [5.0, 2.0, 1.0, 2.0],
                    ]
                ],
                [
                    [
                        [3.0, 4.0, 1.0, 2.0],
                        [6.0, 3.0, 1.0, 2.0],
                        [6.0, 5.0, 4.0, 1.0],
                        [7.0, 7.0, 9.0, 7.0],
                    ]
                ],
                [
                    [
                        [1.0, 1.0, 1.0, 2.0],
                        [1.0, 1.0, 1.0, 2.0],
                        [8.0, 0.0, 4.0, 6.0],
                        [0.0, 0.0, 3.0, 2.0],
                    ]
                ],
            ]
        )

    @pytest.fixture
    def targets(self):
        return torch.Tensor([1, 2, 3, 0]).to(dtype=torch.long)

    def test_mixup(self, inputs, targets):
        mixup = MixUpCutMix(
            augment_prob=1, mixup_alpha=1, switch_prob=0, cutmix_alpha=0, classes=4
        )
        actual_images, actual_targets = mixup(images=inputs, targets=targets)
        assert_expected(
            actual_images,
            torch.Tensor(
                (
                    [
                        [
                            [
                                [1.0000, 1.5626, 1.0000, 2.0000],
                                [1.5626, 1.0000, 1.0000, 2.0000],
                                [4.6245, 1.1252, 2.3123, 2.6245],
                                [3.9381, 2.8129, 3.5626, 5.9381],
                            ]
                        ],
                        [
                            [
                                [2.4374, 4.5626, 1.0000, 2.0000],
                                [3.1871, 4.1252, 1.0000, 2.0000],
                                [4.3123, 2.7497, 6.8129, 1.5626],
                                [5.8748, 4.1871, 4.4993, 4.1871],
                            ]
                        ],
                        [
                            [
                                [2.5626, 4.4374, 1.0000, 2.0000],
                                [3.8129, 3.8748, 1.0000, 2.0000],
                                [4.6877, 3.2503, 6.1871, 1.4374],
                                [6.1252, 4.8129, 5.5007, 4.8129],
                            ]
                        ],
                        [
                            [
                                [1.0000, 1.4374, 1.0000, 2.0000],
                                [1.4374, 1.0000, 1.0000, 2.0000],
                                [5.3755, 0.8748, 2.6877, 3.3755],
                                [3.0619, 2.1871, 3.4374, 5.0619],
                            ]
                        ],
                    ]
                )
            ),
            atol=0.0001,
            rtol=0.0,
        )
        assert_expected(
            actual_targets,
            torch.Tensor(
                [
                    [0.4187, 0.5313, 0.0250, 0.0250],
                    [0.0250, 0.0250, 0.5313, 0.4187],
                    [0.0250, 0.0250, 0.4187, 0.5313],
                    [0.5313, 0.4187, 0.0250, 0.0250],
                ]
            ),
            atol=0.0001,
            rtol=0.0,
        )

    def test_cutmix(self, inputs, targets):
        cutmix = MixUpCutMix(
            augment_prob=1, mixup_alpha=1, switch_prob=1, cutmix_alpha=1, classes=4
        )
        actual_images, actual_targets = cutmix(images=inputs, targets=targets)
        assert_expected(
            actual_images,
            torch.Tensor(
                (
                    [
                        [
                            [
                                [1.0, 2.0, 1.0, 2.0],
                                [2.0, 1.0, 1.0, 2.0],
                                [8.0, 2.0, 1.0, 0.0],
                                [0.0, 5.0, 4.0, 9.0],
                            ]
                        ],
                        [
                            [
                                [2.0, 5.0, 1.0, 2.0],
                                [1.0, 5.0, 1.0, 2.0],
                                [6.0, 1.0, 9.0, 2.0],
                                [7.0, 2.0, 1.0, 2.0],
                            ]
                        ],
                        [
                            [
                                [3.0, 4.0, 1.0, 2.0],
                                [6.0, 3.0, 1.0, 2.0],
                                [3.0, 5.0, 4.0, 1.0],
                                [5.0, 7.0, 9.0, 7.0],
                            ]
                        ],
                        [
                            [
                                [1.0, 1.0, 1.0, 2.0],
                                [1.0, 1.0, 1.0, 2.0],
                                [2.0, 0.0, 4.0, 6.0],
                                [7.0, 0.0, 3.0, 2.0],
                            ]
                        ],
                    ]
                ),
            ),
            atol=0.0001,
            rtol=0.0,
        )
        assert_expected(
            actual_targets,
            torch.Tensor(
                [
                    [0.1375, 0.8125, 0.0250, 0.0250],
                    [0.0250, 0.0250, 0.8125, 0.1375],
                    [0.0250, 0.0250, 0.1375, 0.8125],
                    [0.8125, 0.1375, 0.0250, 0.0250],
                ]
            ),
            atol=0.0001,
            rtol=0.0,
        )

    def test_no_augment(self, inputs, targets):
        cutmix = MixUpCutMix(
            augment_prob=0, mixup_alpha=1, switch_prob=1, cutmix_alpha=1, classes=4
        )
        actual_images, actual_targets = cutmix(images=inputs, targets=targets)
        assert_expected(actual_images, inputs)
        assert_expected(
            actual_targets,
            torch.Tensor(
                [
                    [0.0250, 0.9250, 0.0250, 0.0250],
                    [0.0250, 0.0250, 0.9250, 0.0250],
                    [0.0250, 0.0250, 0.0250, 0.9250],
                    [0.9250, 0.0250, 0.0250, 0.0250],
                ]
            ),
            atol=0.0001,
            rtol=0.0,
        )


@pytest.fixture
def wav():
    with open(get_asset_path("kaldi_file_8000.wav"), "rb") as f:
        wav = f.read()
    out = torch.frombuffer(wav, dtype=torch.uint8)
    return out


class TestAudioEvalTransform:
    @pytest.fixture
    def transform(self):
        return AudioEvalTransform()

    def test_transform(self, transform, wav):
        actual = transform(wav)
        assert_expected(actual.size(), (1, 1024, 128))
        assert_expected(actual.sum().item(), 52000.8828, atol=0.0001, rtol=0.0)

    def test_transform_list(self, transform, wav):
        actual = transform([wav])
        assert_expected(actual.size(), (1, 1, 1024, 128))
        assert_expected(actual.sum().item(), 52000.8828, atol=0.0001, rtol=0.0)


class TestAudioPretrainTransform:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        np.random.seed(0)

    @pytest.fixture
    def transform(self):
        return AudioPretrainTransform()

    def test_transform(self, transform, wav):
        actual = transform(wav)
        assert_expected(actual.size(), (1, 1024, 128))
        assert_expected(actual.sum().item(), 52072.4531, atol=0.0001, rtol=0.0001)

    def test_transform_list(self, transform, wav):
        actual = transform([wav])
        assert_expected(actual.size(), (1, 1, 1024, 128))
        assert_expected(actual.sum().item(), 52072.4531, atol=0.0001, rtol=0.0001)


class TestAudioFinetuneTransform:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        set_rng_seed(0)
        np.random.seed(0)

    @pytest.fixture
    def transform(self):
        return AudioFineTuneTransform()

    def test_transform(self, transform, wav):
        actual = transform(wav)
        assert_expected(actual.size(), (1, 1024, 128))
        assert_expected(actual.sum().item(), 53656.75, atol=0.0001, rtol=0.0001)

    def test_transform_list(self, transform, wav):
        actual = transform([wav])
        assert_expected(actual.size(), (1, 1, 1024, 128))
        assert_expected(actual.sum().item(), 53656.75, atol=0.0001, rtol=0.0001)

    def test_transform_with_mixup(self, transform, wav):
        with open(get_asset_path("sinewave.wav"), "rb") as f:
            bfr = f.read()
        mixup_wav = [torch.frombuffer(bfr, dtype=torch.uint8)]
        actual = transform(wav, mixup_wav, mix_lambda=0.5)
        assert_expected(actual.sum().item(), 54631.046875, atol=0.0001, rtol=0.0001)


class TestRandAug:
    @pytest.fixture(autouse=True)
    def set_seed(self):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    @pytest.fixture
    def inputs(self):
        img = torch.Tensor(
            [
                [
                    [1.0, 2.0, 1.0, 2.0],
                    [2.0, 1.0, 1.0, 2.0],
                    [2.0, 2.0, 1.0, 0.0],
                    [7.0, 5.0, 4.0, 9.0],
                ],
                [
                    [1.0, 2.0, 1.0, 2.0],
                    [2.0, 1.0, 1.0, 2.0],
                    [2.0, 2.0, 1.0, 0.0],
                    [7.0, 5.0, 4.0, 9.0],
                ],
                [
                    [1.0, 2.0, 1.0, 2.0],
                    [2.0, 1.0, 1.0, 2.0],
                    [2.0, 2.0, 1.0, 0.0],
                    [7.0, 5.0, 4.0, 9.0],
                ],
            ]
        )
        return transforms.ToPILImage()(img)

    def test_all_augment(self, inputs):
        aug = RandAug(num_ops=15, prob=1, sample_with_replacement=False)
        actual_img = aug(inputs)
        torch.testing.assert_close(
            transforms.ToTensor()(actual_img),
            torch.Tensor(
                [
                    [
                        [0.7490, 0.0118, 0.3373, 0.8745],
                        [0.5333, 0.0667, 0.4667, 0.0275],
                        [0.5098, 0.0510, 0.5608, 0.6627],
                        [0.8039, 0.0000, 0.4863, 0.4863],
                    ],
                    [
                        [0.7843, 0.0118, 0.3294, 0.8784],
                        [0.5490, 0.0627, 0.4667, 0.0235],
                        [0.5294, 0.0471, 0.5647, 0.6941],
                        [0.8392, 0.0000, 0.4549, 0.4549],
                    ],
                    [
                        [0.2000, 0.0118, 0.3294, 0.7882],
                        [0.6667, 0.0667, 0.4667, 0.0980],
                        [0.4784, 0.0471, 0.5608, 0.1098],
                        [0.1020, 0.0000, 0.4078, 0.4078],
                    ],
                ]
            ),
            atol=0.0001,
            rtol=0.0,
        )
