# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from io import BytesIO
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import torch
import torchaudio
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F


def _cut_pad_waveform_pair(
    waveform1: Tensor, waveform2: Tensor
) -> Tuple[Tensor, Tensor]:
    if waveform1.shape[1] != waveform2.shape[1]:
        if waveform1.shape[1] > waveform2.shape[1]:
            # padding
            temp_wav = torch.zeros(1, waveform1.shape[1])
            temp_wav[0, 0 : waveform2.shape[1]] = waveform2
            waveform2 = temp_wav
        else:
            # cutting
            waveform2 = waveform2[0, 0 : waveform1.shape[1]]

    return waveform1, waveform2


class ImageEvalTransform:
    """
    Standard image transform for MAE eval.
    Args:
        input_size (int): Input image size. Default is 224.
        interpolation (int): Interpolation method for resizing. Default is bicubic.
        mean (Tuple[float, float, float]): mean for normalization. Default is imagenet mean (0.485, 0.456, 0.406).
        std (Tuple[float, float, float]): std for normalization. Default is imagenet std (0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        input_size: int,
        interpolation: int = Image.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)
        img_transforms: List[Callable] = [
            transforms.Resize(size, interpolation=interpolation),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.img_transforms = transforms.Compose(img_transforms)

    def __call__(self, image: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Args:
            image (Union[Image.Image, List[Image.Image]]): input pil image or list of pil images
        Returns:
            Tensor: image tensor after applying transforms. collation is done if input is a list.

        """
        if isinstance(image, Image.Image):
            # pyre-fixme[7]: Expected `Tensor` but got `Image`.
            return self.img_transforms(image)
        img_tensors = []
        for img in image:
            img_tensors.append(self.img_transforms(img))
        return torch.stack(img_tensors)


class ImagePretrainTransform:
    """
    Standard image transform for MAE pretraining.
    Args:
        input_size (int): Input image size. Default is 224.
        scale (Tuple[float, float]): Scale for resizing. Default is (0.2, 1.0
        interpolation (int): Interpolation method for resizing. Default is bicubic.
        mean (Tuple[float, float, float]): mean for normalization. Default is imagenet mean (0.485, 0.456, 0.406).
        std (Tuple[float, float, float]): std for normalization. Default is imagenet std (0.229, 0.224, 0.225).
    """

    def __init__(
        self,
        input_size: int,
        scale: Tuple[float, float] = (0.2, 1.0),
        interpolation: int = Image.BICUBIC,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:

        img_transforms: List[Callable] = [
            transforms.RandomResizedCrop(
                input_size, scale=scale, interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.img_transforms = transforms.Compose(img_transforms)

    def __call__(self, image: Union[Image.Image, List[Image.Image]]) -> Tensor:
        """
        Args:
            image (Union[Image.Image, List[Image.Image]]): input pil image or list of pil images
        Returns:
            Tensor: image tensor after applying transforms. collation is done if input is a list.

        """
        if isinstance(image, Image.Image):
            # pyre-fixme[7]: Expected `Tensor` but got `Image`.
            return self.img_transforms(image)
        img_tensors = []
        for img in image:
            img_tensors.append(self.img_transforms(img))
        return torch.stack(img_tensors)


class MixUpCutMix:
    """
    Augment batch of images and labels using mixup or cutmix depending on a probability at a batch level.
    The code is adapted from timm version https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py#L90
    Args:
        augment_prob (float): Probability of applying augmentation. Default is 1.0.
        mixup_alpha (float): Mixup alpha. Default is 0.8.
        cutmix_alpha (float): Cutmix alpha. Default is 1.0
        switch_prob (float): Probability of using cutmix instead of mixup. Default is 0.5.
        classes (int): Number of classes in labels. Default is 1000.
        label_smoothing (float): Label smoothing factor. Default is 0.1.

    """

    def __init__(
        self,
        augment_prob: float = 1.0,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        switch_prob: float = 0.5,
        classes: int = 1000,
        label_smoothing: float = 0.1,
    ):
        self.augment_prob = augment_prob
        if mixup_alpha > 0 and cutmix_alpha > 0:
            if switch_prob == 0.0:
                raise ValueError(
                    "switch_prob must be > 0 if mixup_alpha and cutmix_alpha > 0."
                )
        elif mixup_alpha > 0 or cutmix_alpha > 0:
            if switch_prob != 0.0:
                raise ValueError(
                    "switch prob must be 0 if only one of mixup_alpha or cutmix_alpha > 0."
                )
        else:
            raise ValueError("mixup_alpha or cutmix_alpha must be > 0.")

        self.mixup_alpha = mixup_alpha
        self.switch_prob = switch_prob
        self.cutmix_alpha = cutmix_alpha
        self.classes = classes
        self.label_smoothing = label_smoothing

    def _get_lambda(self) -> Tuple[float, bool]:
        lam: float = 1
        use_cutmix = False
        if np.random.rand() < self.augment_prob:
            if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
                use_cutmix = np.random.rand() < self.switch_prob
            elif self.mixup_alpha > 0:
                use_cutmix = False
            elif self.cutmix_alpha > 0:
                use_cutmix = True

            if use_cutmix:
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return lam, use_cutmix

    def _get_cutmix_bbox(self, images: Tensor, lam: float) -> Tuple[int, int, int, int]:
        _, _, h, w = images.size()
        ratio = np.sqrt(1 - lam)
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cut_center_y = np.random.randint(0, h)
        cut_center_x = np.random.randint(0, w)

        cut_y_up = np.clip(cut_center_y - cut_h // 2, 0, h)
        cut_y_down = np.clip(cut_center_y + cut_h // 2, 0, h)
        cut_x_left = np.clip(cut_center_x - cut_w // 2, 0, w)
        cut_x_right = np.clip(cut_center_x + cut_w // 2, 0, w)

        return (cut_y_up, cut_y_down, cut_x_left, cut_x_right)

    def _get_smoothed_label_prob(self, targets: Tensor) -> Tensor:
        bsz = targets.size(0)
        # non label value = smoothing / classes, label value = 1 - smoothing/classes * (classes-1)
        non_label_prob = self.label_smoothing / self.classes
        label_prob = 1 + non_label_prob - self.label_smoothing
        labels = (
            torch.full((bsz, self.classes), non_label_prob)
            .to(device=targets.device)
            .scatter_(1, targets.view(-1, 1), label_prob)
        )
        return labels

    def __call__(self, images: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        lam, use_cutmix = self._get_lambda()
        if lam != 1.0:
            if use_cutmix:
                cut_y_up, cut_y_down, cut_x_left, cut_x_right = self._get_cutmix_bbox(
                    images, lam
                )
                bbox_area = (cut_y_down - cut_y_up) * (cut_x_right - cut_x_left)
                _, _, h, w = images.size()
                lam = 1.0 - bbox_area / float(h * w)
                images[:, :, cut_y_up:cut_y_down, cut_x_left:cut_x_right] = images.flip(
                    0
                )[:, :, cut_y_up:cut_y_down, cut_x_left:cut_x_right]
            else:
                flipped_images = images.flip(0).mul_(1 - lam)
                images.mul_(lam).add_(flipped_images)

        y1 = self._get_smoothed_label_prob(targets)
        y2 = self._get_smoothed_label_prob(targets.flip(0))
        targets = y1 * lam + y2 * (1 - lam)
        return images, targets


class RandAug:
    """
    MAE specific variant of RandAug for images as described in https://arxiv.org/pdf/1909.13719v2.pdf. Code adapted from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/auto_augment.py#L736

    Args:
        num_ops (int): Number of operations to perform. Defaults to 2.
        magnitude (int): Magnitude of the operation. Defaults to 9.
        prob (float): Probability of applying the operation. Defaults to 0.5.
        magnitude_std (float): Std deviation of the magnitude. Defaults to 0.5
        sample_with_replacement (bool): Whether to sample with replacement or not. Defaults to True.
    """

    MAX_MAG = 10
    FILL_COLOR = (124, 116, 104)
    INTERPOLATIONS = (Image.BILINEAR, Image.BICUBIC)

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        prob: float = 0.5,
        magnitude_std: float = 0.5,
        sample_with_replacement: bool = True,
    ) -> None:
        self.magnitude = magnitude
        self.ops = [
            "AutoContrast",
            "Equalize",
            "Invert",
            "Rotate",
            "PosterizeIncreasing",
            "SolarizeIncreasing",
            "SolarizeAdd",
            "ColorIncreasing",
            "ContrastIncreasing",
            "BrightnessIncreasing",
            "SharpnessIncreasing",
            "ShearX",
            "ShearY",
            "TranslateXRel",
            "TranslateYRel",
        ]
        self.num_ops = num_ops
        self.prob = prob
        self.magnitude_std = magnitude_std
        self.sample_with_replacement = sample_with_replacement

    def _randomly_negate(self, v: float) -> float:
        return -v if random.random() > 0.5 else v

    def _solarize_add(
        self, img: Image.Image, add: int, thresh: int = 128
    ) -> Image.Image:
        if img.mode in ("L", "RGB"):
            lut = []
            for i in range(256):
                if i < thresh:
                    lut.append(min(255, i + add))
                else:
                    lut.append(i)
            if img.mode == "RGB":
                lut = lut + lut + lut
            return img.point(lut)
        else:
            return img

    def __call__(self, x: Image.Image) -> Union[Image.Image, Tensor]:
        """
        Args:
            x (Image.Image): input image
        Returns:
            Union[Image.Image, Tensor]: Pil image after applying the ops. The Union is only meant to make type checker happy

        """
        ops = np.random.choice(
            self.ops, self.num_ops, replace=self.sample_with_replacement
        )
        for op in ops:
            if random.random() > self.prob:
                continue
            if self.magnitude_std > 0:
                magnitude = random.gauss(self.magnitude, self.magnitude_std)
            else:
                magnitude = self.magnitude
            magnitude = min(self.MAX_MAG, max(0, magnitude))

            if op == "AutoContrast":
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.autocontrast(x)
            elif op == "Equalize":
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.equalize(x)
            elif op == "Invert":
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.invert(x)
            elif op == "Rotate":
                angle = (magnitude / self.MAX_MAG) * 30.0
                angle = self._randomly_negate(angle)
                interpolation = random.choice(self.INTERPOLATIONS)
                x = x.rotate(angle, fillcolor=self.FILL_COLOR, resample=interpolation)
            elif op == "PosterizeIncreasing":
                bits = 4 - int((magnitude / self.MAX_MAG) * 4)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.posterize(img=x, bits=bits)
            elif op == "SolarizeIncreasing":
                threshold = 256 - int((magnitude / self.MAX_MAG) * 256)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.solarize(img=x, threshold=threshold)
            elif op == "SolarizeAdd":
                add = int((magnitude / self.MAX_MAG) * 110)
                x = self._solarize_add(img=x, add=add)
            elif op == "ColorIncreasing":
                saturation_factor = (magnitude / self.MAX_MAG) * 0.9
                saturation_factor = 1.0 + self._randomly_negate(saturation_factor)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.adjust_saturation(img=x, saturation_factor=saturation_factor)
            elif op == "ContrastIncreasing":
                contrast_factor = (magnitude / self.MAX_MAG) * 0.9
                contrast_factor = 1.0 + self._randomly_negate(contrast_factor)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.adjust_contrast(img=x, contrast_factor=contrast_factor)
            elif op == "BrightnessIncreasing":
                brightness_factor = (magnitude / self.MAX_MAG) * 0.9
                brightness_factor = 1.0 + self._randomly_negate(brightness_factor)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.adjust_brightness(img=x, brightness_factor=brightness_factor)
            elif op == "SharpnessIncreasing":
                sharpness_factor = (magnitude / self.MAX_MAG) * 0.9
                sharpness_factor = 1.0 + self._randomly_negate(sharpness_factor)
                # pyre-fixme[9]: x has type `Image`; used as `Tensor`.
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got `Image`.
                x = F.adjust_sharpness(img=x, sharpness_factor=sharpness_factor)
            elif op == "ShearX":
                shear = (magnitude / self.MAX_MAG) * 0.3
                shear = self._randomly_negate(shear)
                interpolation = random.choice(self.INTERPOLATIONS)
                x = x.transform(
                    x.size,
                    Image.AFFINE,
                    (1, shear, 0, 0, 1, 0),
                    fillcolor=self.FILL_COLOR,
                    resample=interpolation,
                )
            elif op == "ShearY":
                shear = (magnitude / self.MAX_MAG) * 0.3
                shear = self._randomly_negate(shear)
                interpolation = random.choice(self.INTERPOLATIONS)
                x = x.transform(
                    x.size,
                    Image.AFFINE,
                    (1, 0, 0, shear, 1, 0),
                    fillcolor=self.FILL_COLOR,
                    resample=interpolation,
                )
            elif op == "TranslateXRel":
                translate = (magnitude / self.MAX_MAG) * 0.45
                translate = self._randomly_negate(translate)
                translate = translate * x.size[0]
                interpolation = random.choice(self.INTERPOLATIONS)
                x = x.transform(
                    x.size,
                    Image.AFFINE,
                    (1, 0, translate, 0, 1, 0),
                    fillcolor=self.FILL_COLOR,
                    resample=interpolation,
                )

            elif op == "TranslateYRel":
                translate = (magnitude / self.MAX_MAG) * 0.45
                translate = self._randomly_negate(translate)
                translate = translate * x.size[1]
                interpolation = random.choice(self.INTERPOLATIONS)
                x = x.transform(
                    x.size,
                    Image.AFFINE,
                    (1, 0, 0, 0, 1, translate),
                    fillcolor=self.FILL_COLOR,
                    resample=interpolation,
                )
        return x


def get_waveform(
    audio_bytes: Tensor, mean_normalize: bool = True
) -> Tuple[Tensor, float]:
    """
    Get waveform and sampling rate from input audio bytes.
    Args:
        audio_bytes (Tensor): Audio bytes tensor.
    Returns:
        Tuple with the waveform and the sampling rate.

    """
    buff = BytesIO(audio_bytes.numpy().tobytes())
    buff.seek(0)
    waveform, sampling_rate = torchaudio.load(buff)
    if mean_normalize:
        waveform = waveform - waveform.mean()
    return waveform, sampling_rate


def roll_mag_aug(waveform: Tensor, alpha: float = 10, beta: float = 10) -> Tensor:
    """
    Samples random starting points and rolls cyclically along the time axis
    Code taken from https://github.com/facebookresearch/AudioMAE/blob/main/dataset.py#L169
    Args:
        waveform (Tensor): Waveform tensor
        alpha (float): alpha for sampling
        beta (float): beta for sampling
    Returns:
        Rolled waveform tensor
    """
    waveform = waveform.numpy()
    idx = np.random.randint(len(waveform))
    rolled_waveform = np.roll(waveform, idx)
    mag = np.random.beta(alpha, beta) + 0.5
    return torch.Tensor(rolled_waveform * mag)


def get_fbank(
    waveform: Tensor,
    sampling_rate: float,
    melbins: int,
    target_length: int,
    mean: float = -4.2677393,
    std: float = 4.5689974,
    freq_mask: int = 0,
    time_mask: int = 0,
) -> Tensor:
    """
    Frequency bank from waveform. Also does normalization and optionally applies rolling augmentation, frequency and time masking.
    Args:
        waveform (Tensor): Waveform tensor
        sampling_rate (float): Sampling rate of the waveform
        melbins (int): Melbins
        target_length (int): Target length of the spectrogram
        mean (float): mean used for normalization. Default is -4.2677393
        std (float): standard deviation used for normalization. Default us 4.5689974
        roll_mag (bool): If True, apply the rolling augmentation. Default False.
        freq_mask (int): Frequency mask to apply to the waveform. Default 0.
        time_mask (int): Time mask to apply to the waveform. Default 0.
    Returns:
        Frequency bank tensor with shape 1 x target_length x melbins
    """

    # fbank shape : frames x melbins
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sampling_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=melbins,
        dither=0.0,
        frame_shift=10,
    )

    n_frames = fbank.shape[0]
    pad = target_length - n_frames

    # fbank shape : target_length x melbins
    if pad > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, pad))
        fbank = m(fbank)
    elif pad < 0:
        fbank = fbank[0:target_length, :]

    fbank = fbank.transpose(0, 1).unsqueeze(0)
    if freq_mask > 0:
        fbank = torchaudio.transforms.FrequencyMasking(freq_mask)(fbank)
    if time_mask > 0:
        fbank = torchaudio.transforms.TimeMasking(time_mask)(fbank)
    fbank = fbank.squeeze(0).transpose(0, 1)

    fbank = (fbank - mean) / (std * 2)
    # Add channel dim
    return fbank.unsqueeze(0)


class AudioEvalTransform:
    """
    Standard Audio MAE eval transform for AudioSet.
    Args:
        melbins (int): Melbins. Defaults to 128.
        target_length (int): Target length of the spectrogram. Defaults to 1024.
        mean (float): mean used for normalization. Default is -4.2677393
        std (float): standard deviation used for normalization. Default us 4.5689974

    """

    def __init__(
        self,
        melbins: int = 128,
        target_length: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
    ) -> None:
        self.melbins = melbins
        self.target_length = target_length
        self.mean = mean
        self.std = std

    def __call__(self, audio_byte_tensor: Union[Tensor, List[Tensor]]) -> Tensor:
        """
        Args:
            audio_byte_tensor (Union[Tensor, List[Tensor]]): An audio bytes tensor or list of tensors.
        Returns:
            Tensor: audio tensor after applying transforms. collation is done if input is a list.
        """

        if isinstance(audio_byte_tensor, List):
            inputs = audio_byte_tensor
            collate = True
        else:
            inputs = [audio_byte_tensor]
            collate = False
        fbanks = []
        for byte_tensor in inputs:
            waveform, sampling_rate = get_waveform(byte_tensor)
            fbank = get_fbank(
                waveform=waveform,
                sampling_rate=sampling_rate,
                melbins=self.melbins,
                target_length=self.target_length,
                mean=self.mean,
                std=self.std,
            )
            fbanks.append(fbank)
        if collate:
            return torch.stack(fbanks, dim=0)

        return fbanks[0]


class AudioPretrainTransform:
    """
    Standard Audio MAE pretrain transform for AudioSet.
    Args:
        melbins (int): Melbins. Defaults to 128.
        target_length (int): Target length of the spectrogram. Defaults to 1024.
        mean (float): mean used for normalization. Default is -4.2677393
        std (float): standard deviation used for normalization. Default us 4.5689974
        roll_mag (bool): If True, apply the rolling augmentation. Default True.
    """

    def __init__(
        self,
        melbins: int = 128,
        target_length: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        roll_mag: bool = True,
    ):
        self.melbins = melbins
        self.target_length = target_length
        self.mean = mean
        self.std = std
        self.roll_mag = roll_mag

    def __call__(self, audio_byte_tensor: Union[Tensor, List[Tensor]]) -> Tensor:
        """
        Args:
            audio_byte_tensor (Union[Tensor, List[Tensor]]): An audio bytes tensor or list of tensors.
        Returns:
            Tensor: audio tensor after applying transforms. collation is done if input is a list.
        """
        if isinstance(audio_byte_tensor, List):
            inputs = audio_byte_tensor
            collate = True
        else:
            inputs = [audio_byte_tensor]
            collate = False
        fbanks = []
        for byte_tensor in inputs:
            waveform, sampling_rate = get_waveform(byte_tensor)
            if self.roll_mag:
                waveform = roll_mag_aug(waveform)
            fbank = get_fbank(
                waveform=waveform,
                sampling_rate=sampling_rate,
                melbins=self.melbins,
                target_length=self.target_length,
                mean=self.mean,
                std=self.std,
            )
            fbanks.append(fbank)

        if collate:
            return torch.stack(fbanks, dim=0)

        return fbanks[0]


class AudioFineTuneTransform:
    """
    Standard Audio MAE finetune transform for AudioSet.
    Args:
        melbins (int): Melbins. Defaults to 128.
        target_length (int): Target length of the spectrogram. Defaults to 1024.
        mean (float): mean used for normalization. Default is -4.2677393
        std (float): standard deviation used for normalization. Default us 4.5689974
        roll_mag (bool): If True, apply the rolling augmentation. Default True.
        freq_mask (int): Frequency mask to apply to the waveform. Default 48.
        time_mask (int): Time mask to apply to the waveform. Default 192.
    """

    def __init__(
        self,
        melbins: int = 128,
        target_length: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        roll_mag: bool = True,
        freq_mask: int = 48,
        time_mask: int = 192,
    ):
        self.melbins = melbins
        self.target_length = target_length
        self.mean = mean
        self.std = std
        self.roll_mag = roll_mag
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __call__(
        self,
        audio_byte_tensor: Union[Tensor, List[Tensor]],
        mixup_audio_byte_tensor: Optional[List[Tensor]] = None,
        mix_lambda: float = -1,
    ) -> Tensor:
        """
        Args:
            audio_byte_tensor (Union[Tensor, List[Tensor]]): An audio bytes tensor or list of tensors.
        Returns:
            Tensor: audio tensor after applying transforms. collation is done if input is a list.
        """

        mixup_inputs = (
            [] if mixup_audio_byte_tensor is None else mixup_audio_byte_tensor
        )

        mixup_enabled = 0 < mix_lambda < 1

        if len(mixup_inputs) > 0 and not mixup_enabled:
            raise ValueError(
                f"When passing mixup inputs mix_lambda must be between 0 and 1, received {mix_lambda}"
            )

        if len(mixup_inputs) == 0 and mixup_enabled:
            raise ValueError("Cannot perform mixup, received empty mixup_inputs")

        if isinstance(audio_byte_tensor, List):
            inputs = audio_byte_tensor
            collate = True
        else:
            inputs = [audio_byte_tensor]
            collate = False

        if mixup_enabled and len(inputs) != len(mixup_inputs):
            raise ValueError("Mixup inputs must have the same length as inputs.")

        fbanks = []
        for i, byte_tensor in enumerate(inputs):
            waveform, sampling_rate = get_waveform(byte_tensor)
            if self.roll_mag:
                waveform = roll_mag_aug(waveform)
            mixup_byte_tensor = mixup_inputs[i] if len(mixup_inputs) > 0 else None
            if mixup_byte_tensor is not None:
                mixup_waveform, mixup_sampling_rate = get_waveform(mixup_byte_tensor)
                if self.roll_mag:
                    mixup_waveform = roll_mag_aug(mixup_waveform)
                waveform, mixup_waveform = _cut_pad_waveform_pair(
                    waveform, mixup_waveform
                )
                waveform = waveform * mix_lambda + mixup_waveform * (1 - mix_lambda)
                waveform = waveform - waveform.mean()
            fbank = get_fbank(
                waveform=waveform,
                sampling_rate=sampling_rate,
                melbins=self.melbins,
                target_length=self.target_length,
                mean=self.mean,
                std=self.std,
                freq_mask=self.freq_mask,
                time_mask=self.time_mask,
            )
            fbanks.append(fbank)

        if collate:
            return torch.stack(fbanks, dim=0)

        return fbanks[0]
