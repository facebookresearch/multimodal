# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import Optional

import torch
from torch import nn, Tensor


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(
        self, pixel_values: Tensor, interpolate_pos_encoding: bool = False
    ) -> Tensor:
        _, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
                )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


class ImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        hidden_dropout_prob: float = 0.0,
        use_image_masking: bool = True,
    ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_size)
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None

    def interpolate_pos_encoding(
        self, embeddings: Tensor, height: int, width: int
    ) -> Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        npatch = embeddings.shape[1] - 1
        n = self.position_embeddings.shape[1] - 1
        if npatch == n and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.patch_size[0]
        w0 = width // self.patch_embeddings.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(n)), int(math.sqrt(n)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(n), w0 / math.sqrt(n)),
            mode="bicubic",
            align_corners=False,
        )
        assert (
            int(h0) == patch_pos_embed.shape[-2]
            and int(w0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: Tensor,
        image_patches_mask: Optional[Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

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
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
