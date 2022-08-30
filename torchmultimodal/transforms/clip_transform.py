# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterable, List, Optional, Tuple, Union

import torch
from PIL.Image import Image
from torch import Tensor
from torchmultimodal import _PATH_MANAGER
from torchtext import transforms as text_transforms
from torchtext.transforms import CLIPTokenizer
from torchvision import transforms as image_transforms
from torchvision.transforms import InterpolationMode

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_DEFAULT_VOCAB_BPE_PATH = "http://download.pytorch.org/models/text/clip_merges.bpe"


def convert_to_rgb(img: Image) -> Image:
    return img.convert("RGB")


class CLIPTextTransform:
    """CLIP text transform
    CLIP BPE tokenizer transform, adds start and end tokens, then pads/truncates to text_max_length as necessary.
    Args:
        text_max_length (int): Maximum length of text token sequences.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        text_bpe_merges_path (str): Location of BPE merges file for text transform.
        text_encoder_json_path (str, optional): Location of BPE encoder JSON file.
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        text (Union[Iterable[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(
        self,
        text_max_length: int = 77,
        text_start_token: str = "<|startoftext|>",
        text_end_token: str = "<|endoftext|>",
        text_bpe_merges_path: str = CLIP_DEFAULT_VOCAB_BPE_PATH,
        text_encoder_json_path: Optional[str] = None,
        num_merges: Optional[int] = 48894,
    ) -> None:

        local_merges_path = _PATH_MANAGER.get_local_path(text_bpe_merges_path)
        tokenizer = CLIPTokenizer(
            local_merges_path, text_encoder_json_path, num_merges=num_merges
        )
        text_start_token = tokenizer([text_start_token])[0][0]
        text_end_token = tokenizer([text_end_token])[0][0]
        text_max_length = text_max_length

        self.text_transform = text_transforms.Sequential(
            *[
                tokenizer,
                text_transforms.Truncate(text_max_length - 2),
                text_transforms.AddToken(text_start_token, begin=True),
                text_transforms.AddToken(text_end_token, begin=False),
                text_transforms.StrToIntTransform(),
                text_transforms.ToTensor(padding_value=0),
                text_transforms.PadTransform(max_length=text_max_length, pad_value=0),
            ]
        )

    def __call__(self, text: Union[Iterable[str], str]) -> Tensor:
        if isinstance(text, str):
            text = [text]
        text_result = self.text_transform(text)
        return text_result


class CLIPImageTransform:
    """CLIP image transform
    random resized crop (train mode) or resize and center crop, followed by RGB conversion, tensor conversion, and normalization.

    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        is_train (bool): Whether transform is run in train mode.

    Inputs:
        image (Union[Iterable[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        image_mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        is_train: bool = True,
    ) -> None:
        joint_transforms: List[Callable] = [
            convert_to_rgb,
            image_transforms.ToTensor(),
            image_transforms.Normalize(image_mean, image_std),
        ]
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        base_transform: List[Callable]
        if is_train:
            base_transform = [
                image_transforms.RandomResizedCrop(
                    image_size, interpolation=image_interpolation
                )
            ]
        else:
            base_transform = [
                image_transforms.Resize(image_size, interpolation=image_interpolation),
                image_transforms.CenterCrop(image_size),
            ]
        self.image_transform = image_transforms.Compose(
            base_transform + joint_transforms
        )

    def __call__(self, image: Union[Iterable[Image], Image]) -> Tensor:
        if isinstance(image, Image):
            image = [image]
        image_result = torch.stack([self.image_transform(x) for x in image])
        return image_result


class CLIPTransform:
    """Image and text transform for CLIP model.

    Image transform: either random resized crop (train mode) or resize and center
        crop, followed by RGB conversion, tensor conversion, and normalization.
    Text transform: applies CLIP's BPE tokenizer transform, adds start and end
        tokens, then pads/truncates to text_max_length as necessary.


    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        text_max_length (int): Maximum length of text token sequences.
        is_train (bool): Whether transform is run in train mode.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        text_bpe_merges_path (str): Location of BPE merges file for text transform.
        text_encoder_json_path (str, optional): Location of BPE encoder JSON file.
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        image (Union[Iterable[Image], Image]): Image or batch of images upon which
            to apply the transform.
        text (Union[Iterable[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (224, 224),
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        image_mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        text_max_length: int = 77,
        is_train: bool = True,
        text_start_token: str = "<|startoftext|>",
        text_end_token: str = "<|endoftext|>",
        text_bpe_merges_path: str = CLIP_DEFAULT_VOCAB_BPE_PATH,
        text_encoder_json_path: Optional[str] = None,
        num_merges: Optional[int] = 48894,
    ) -> None:

        self.image_transform = CLIPImageTransform(
            image_size, image_interpolation, image_mean, image_std, is_train
        )
        self.text_transform = CLIPTextTransform(
            text_max_length,
            text_start_token,
            text_end_token,
            text_bpe_merges_path,
            text_encoder_json_path,
            num_merges,
        )

    def __call__(
        self, image: Union[Iterable[Image], Image], text: Union[Iterable[str], str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_transform(image), self.text_transform(text)
