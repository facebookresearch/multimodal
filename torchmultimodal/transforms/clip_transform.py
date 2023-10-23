# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import html
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import ftfy
import regex as re

import torch
from PIL.Image import Image
from torch import nn, Tensor
from torchmultimodal import _PATH_MANAGER
from torchmultimodal.transforms import text_transforms
from torchvision import transforms as image_transforms
from torchvision.transforms import InterpolationMode


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_DEFAULT_VOCAB_BPE_PATH = "http://download.pytorch.org/models/text/clip_merges.bpe"


def convert_to_rgb(img: Image) -> Image:
    return img.convert("RGB")


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))  # noqa


def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class CLIPBPETokenizer:
    """
    Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

    This implementation is adapted from https://git.io/JDTuJ.
    Example usage:
    tokenizer = CLIPBPETokenizer()
    sentence = "Hello I am using CLIP tokenizer."
    tokens = tokenizer.encode(sentence)
    tokens -> [3306, 328, 687, 1996, 9289, 32634, 23895, 269]
    decoded_sentence = tokenizer.decode(tokens)
    decoded_sentence -> "hello i am using clip tokenizer ."

    Args:
        bpe_path (str): path to the BPE file
        bos_token (str): beginning of sentence token. Defaults to "<|startoftext|>".
        eos_token (str): end of sentence token. Defaults to "<|endoftext|>".
        num_merges (Optional[int]): number of merges.
            If None, it will load all merges from the BPE file.
    """

    def __init__(
        self,
        bpe_path: str = CLIP_DEFAULT_VOCAB_BPE_PATH,
        bos_token: str = "<|startoftext|>",
        eos_token: str = "<|endoftext|>",
        num_merges: Optional[int] = None,  # TODO: add docstring
    ):
        self._separator = "\u0001"
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with _PATH_MANAGER.open(bpe_path, "r", encoding="utf-8") as f:
            bpe_merges = f.read().split("\n")[1:]
        num_merges = num_merges or len(bpe_merges)
        bpe_merges = bpe_merges[:num_merges]
        self.bpe_merges = bpe_merges[:num_merges]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.num_merges = num_merges
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # noqa
        bpe_vocab = list(bytes_to_unicode().values())
        bpe_vocab = bpe_vocab + [v + "</w>" for v in bpe_vocab]
        bpe_vocab.extend(
            ["".join(merge_pair) for merge_pair in bpe_merges[:num_merges]]
        )
        special_tokens = [bos_token, eos_token]
        bpe_vocab.extend(special_tokens)
        self.bpe_vocab = bpe_vocab
        self.encoder = {v: i for i, v in enumerate(bpe_vocab)}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {tok: tok for tok in special_tokens}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text: str) -> List[int]:
        bpe_tokens: List[int] = []
        text = text.lower().strip()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens: List[int]) -> str:
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


class CLIPBPETransform(nn.Module):
    """
    nn.Module wrapper around CLIPBPETokenizer. Supports either a single string
    or list of strings for tokenization.

    Args:
        bpe_path (Optional[str]): path to the BPE file.
            Defaults to CLIP_DEFAULT_VOCAB_BPE_PATH
        bos_token (Optional[str]): beginning of sentence token.
            Defaults to "<|startoftext|>".
        eos_token (Optional[str]): end of sentence token.
            Defaults to "<|endoftext|>".
        num_merges (Optional[int]): number of merges.
            If None, it will load all merges from the BPE file.
    """

    def __init__(
        self,
        bpe_path: Optional[str] = CLIP_DEFAULT_VOCAB_BPE_PATH,
        bos_token: Optional[str] = "<|startoftext|>",
        eos_token: Optional[str] = "<|endoftext|>",
        num_merges: Optional[int] = None,
    ):
        super().__init__()
        self.bpe = CLIPBPETokenizer(
            bpe_path=bpe_path,
            bos_token=bos_token,
            eos_token=eos_token,
            num_merges=num_merges,
        )

    def forward(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self.bpe.encode(text)
        else:
            return [self.bpe.encode(t) for t in text]


class CLIPTextTransform(nn.Module):
    """CLIP text transform
    CLIP BPE tokenizer transform, adds start and end tokens, then pads/truncates to text_max_length as necessary.
    This transform is torch scriptable.
    Args:
        text_max_length (int): Maximum length of text token sequences.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        text_bpe_merges_path (str): Location of BPE merges file for text transform.
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        text (Union[List[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(
        self,
        text_max_length: int = 77,
        text_start_token: str = "<|startoftext|>",
        text_end_token: str = "<|endoftext|>",
        text_bpe_merges_path: str = CLIP_DEFAULT_VOCAB_BPE_PATH,
        num_merges: Optional[int] = 48894,
    ) -> None:

        super().__init__()
        local_merges_path = _PATH_MANAGER.get_local_path(text_bpe_merges_path)
        tokenizer = CLIPBPETransform(
            local_merges_path, text_start_token, text_end_token, num_merges
        )
        text_start_token = tokenizer([text_start_token])[0][0]
        text_end_token = tokenizer([text_end_token])[0][0]
        text_max_length = text_max_length

        self.text_transform = nn.Sequential(
            *[
                tokenizer,
                text_transforms.Truncate(text_max_length - 2),
                text_transforms.AddToken(text_start_token, begin=True),
                text_transforms.AddToken(text_end_token, begin=False),
                text_transforms.ToTensor(padding_value=0),
                text_transforms.PadTransform(max_length=text_max_length, pad_value=0),
            ]
        )

    def forward(self, text: Union[List[str], str]) -> Tensor:
        text_result = self.text_transform(text)
        assert torch.jit.isinstance(text_result, Tensor)
        return text_result


class CLIPImageTransform(nn.Module):
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
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        image_mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        joint_transforms: List[Callable] = [
            convert_to_rgb,
            image_transforms.ToTensor(),
            image_transforms.Normalize(image_mean, image_std),
        ]
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

    def forward(self, image: Union[List[Image], Image]) -> Tensor:
        if isinstance(image, Image):
            return self.image_transform(image)
        image_result = torch.stack([self.image_transform(x) for x in image])
        return image_result


class CLIPTransform(nn.Module):
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
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
        text (Union[List[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        image_mean: Tuple[float, float, float] = CLIP_DEFAULT_MEAN,
        image_std: Tuple[float, float, float] = CLIP_DEFAULT_STD,
        text_max_length: int = 77,
        is_train: bool = True,
        text_start_token: str = "<|startoftext|>",
        text_end_token: str = "<|endoftext|>",
        text_bpe_merges_path: str = CLIP_DEFAULT_VOCAB_BPE_PATH,
        num_merges: Optional[int] = 48894,
    ) -> None:

        super().__init__()
        self.image_transform = CLIPImageTransform(
            image_size, image_interpolation, image_mean, image_std, is_train
        )
        self.text_transform = CLIPTextTransform(
            text_max_length,
            text_start_token,
            text_end_token,
            text_bpe_merges_path,
            num_merges,
        )

    def forward(
        self, image: Union[List[Image], Image], text: Union[List[str], str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_transform(image), self.text_transform(text)
