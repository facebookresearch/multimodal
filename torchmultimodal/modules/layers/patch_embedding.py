# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import Iterable, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.masking.random_masking import (
    random_masking,
    random_masking_2d,
)


class PatchEmbeddingsOutput(NamedTuple):
    embeddings: Tensor
    random_mask: Optional[Tensor] = None
    ids_restore: Optional[Tensor] = None


class PatchEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings for vision transformer
    Args:
        image_size (Union[int, Tuple[int, int]]): Size of the input. If set to an int, we assume a square input. Defaults to 224.
        patch_size (int): Size of the patch. Defaults to 16.
        num_channels (int): Number of channels in the input. Defaults to 3.
        hidden_size (int): Embedding dimension of the output. Defaults to 768.
        hidden_dropout_prob (float): Dropout probability applied after adding position embeddings. Defaults to 0.0.
        use_image_masking (bool): Whether to use image masking or not. Defaults to False.
        patch_drop_rate (Optional[Union[float, Tuple[float, float]]]): ratio of patches to be masked out
        or dropped if single float. Set to tuple if dimension wise masking is needed i.e. 2d masking
        after adding position embeddings as described in https://arxiv.org/pdf/2212.00794.pdf. Defaults to None.
        include_cls_embed (bool): Whether to include the [CLS] token embedding. Defaults to True.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
        use_image_masking: bool = False,
        patch_drop_rate: Optional[Union[float, Tuple[float, float]]] = None,
        include_cls_embed: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("Image size needs to be divisible by patch size")

        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        num_patches = self.num_patches_h * self.num_patches_w

        self.include_cls_embed = include_cls_embed
        if self.include_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            num_patches = num_patches + 1
        self.conv_projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self._init_conv_weights()

        self.image_size: Tuple[int, int] = image_size

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None

        self.patch_drop_rate = patch_drop_rate

    def _init_conv_weights(self) -> None:
        fan_in = (
            self.conv_projection.in_channels
            * self.conv_projection.kernel_size[0]
            * self.conv_projection.kernel_size[1]
        )
        nn.init.trunc_normal_(self.conv_projection.weight, std=math.sqrt(1 / fan_in))
        assert self.conv_projection.bias is not None
        nn.init.zeros_(self.conv_projection.bias)

    def forward(
        self,
        pixel_values: Tensor,
        image_patches_mask: Optional[Tensor] = None,
    ) -> PatchEmbeddingsOutput:
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match image size \
                {self.image_size[0]}*{self.image_size[1]} expected by model"
            )
        embeddings = self.conv_projection(pixel_values).flatten(2).transpose(1, 2)

        _, seq_len, _ = embeddings.size()
        if image_patches_mask is not None:
            if self.mask_token is not None:
                mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
                # replace the masked visual tokens by mask_tokens
                w = image_patches_mask.unsqueeze(-1).type_as(mask_tokens)
                embeddings = embeddings * (1 - w) + mask_tokens * w
            else:
                warnings.warn(
                    "image_patches_mask passed but use_image_masking in init was false. Ignoring."
                )

        # adding positional embeddings excluding [CLS] token (due to possibility of random masking)
        if self.include_cls_embed:
            embeddings = embeddings + self.position_embeddings[:, 1:, :]
        else:
            embeddings = embeddings + self.position_embeddings
        # apply random masking random masking according to patch_drop_rate
        random_mask = None
        ids_restore = None
        if self.training and self.patch_drop_rate is not None:
            if isinstance(self.patch_drop_rate, Iterable):
                embeddings = random_masking_2d(
                    embeddings,
                    mask_ratio_h=self.patch_drop_rate[0],  # type: ignore
                    mask_ratio_w=self.patch_drop_rate[1],  # type: ignore
                    num_patches_h=self.num_patches_h,
                    num_patches_w=self.num_patches_w,
                )
            else:
                embeddings, random_mask, ids_restore, _ = random_masking(
                    embeddings, mask_ratio=self.patch_drop_rate
                )

        # add the [CLS] token to the embedded patch tokens and its positional embedding
        if self.include_cls_embed:
            assert hasattr(
                self, "cls_token"
            ), "CLS token must be defined to include CLS embedding"
            cls_token = self.cls_token + self.position_embeddings[:, :1, :]
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.dropout(embeddings)

        return PatchEmbeddingsOutput(
            embeddings=embeddings,
            random_mask=random_mask,
            ids_restore=ids_restore,
        )
