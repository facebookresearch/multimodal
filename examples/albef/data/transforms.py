# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Tuple, Union

import torch

from torchtext.transforms import (
    AddToken,
    BERTTokenizer,
    PadTransform,
    Sequential,
    StrToIntTransform,
    ToTensor,
    Truncate,
)
from torchvision import transforms

# mean and standard deviation from the ALBEF repo:
# https://github.com/salesforce/ALBEF/blob/main/dataset/__init__.py#L16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)


class ALBEFTextTransform:
    """
    Remove punctuations and trailing spaces in input text and transform it into
    a Tensor of token ids using BERTTokenizer.

    Args:
        vocab_file (str): Local or URL path to pre-trained vocab file.
            Defaults to HuggingFace BERT base (uncased) model's vocab file.
        do_pre_process (bool): Whether to pre-process input text.
            Defaults to True.
        do_lower_case (bool): Whether to convert input text to lowercase.
            Defaults to True.
        truncate (bool): Whether to truncate input text to max_seq_length.
            Defaults to False.
        pad_to_max_seq_len (bool): Whether to pad the sequence to max_seq_length.
        add_end_token (bool): Whether to add the end-of-sentence token.
            Defaults to True.
        max_seq_len (int): The max sequence length after truncating or padding.
            Defaults to 25.
        cls_token_id (int): Value to represent the start of each text.
            Defaults to 101, Hugging Face's BERT cls token id.
        sep_token_id (int): Value to represent the end of each text.
            Defaults to 102, Hugging Face's BERT sep token id.
        pad_token_id (int): Value with which to pad each text so that all texts are the same length.
            Defaults to 0, Hugging Face's BERT pad token id.

    Inputs:
        text (Union[List[str], str]): Input text to transform.
    """

    def __init__(
        self,
        vocab_file: str = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        do_pre_process: bool = True,
        do_lower_case: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

        # start_token + max_word_tokens + optional(end_token) = max_seq_len
        max_word_tokens = max_seq_len - (2 if add_end_token else 1)

        self.tokenizer = Sequential(
            BERTTokenizer(
                vocab_path=vocab_file, do_lower_case=do_lower_case, return_tokens=False
            ),
            StrToIntTransform(),
            Truncate(max_seq_len=max_word_tokens) if truncate else torch.nn.Identity(),
            AddToken(self.cls_token_id, begin=True),
            AddToken(self.sep_token_id, begin=False)
            if add_end_token
            else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
            if pad_to_max_seq_len
            else torch.nn.Identity(),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        input_ids = self.tokenizer(text)
        return input_ids


def training_image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )


def testing_image_transform(
    image_size: int = 384,
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=image_interpolation
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )
