# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Tuple

import torch
from torch import Tensor


class RandomMaskingOutput(NamedTuple):
    x_masked: Tensor
    mask: Tensor
    ids_restore: Tensor
    ids_keep: Tensor


def random_masking(
    x: torch.Tensor,
    mask_ratio: float,
) -> RandomMaskingOutput:
    """
    Original paper: https://arxiv.org/pdf/2111.06377.pdf
    OSS implementation: https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    n, l, d = x.shape  # batch, length, dim
    len_keep = int(l * (1 - mask_ratio))

    noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

    assert len_keep >= 1, "must keep at least 1 patch"

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([n, l], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return RandomMaskingOutput(
        x_masked=x_masked,
        mask=mask,
        ids_restore=ids_restore,
        ids_keep=ids_keep,
    )


def random_masking_2d(
    x: torch.Tensor,
    mask_ratio_h: float,
    mask_ratio_w: float,
    num_patches_h: int,
    num_patches_w: int,
) -> Tensor:
    """
    Perform 2d masking as described in audio mae paper https://arxiv.org/pdf/2207.06405.pdf
    Code adapted from https://github.com/facebookresearch/AudioMAE/blob/main/models_vit.py#L88
    Args:
        x: Input tensor containing patches of shape bsz x seq_len x embed_dim
        mask_ratio_h: masking ratio for height dimension
        mask_ratio_w: masking ratio for width dimension
        num_patches_h: number of patches in height dimension
        num_patches_w: number of patches in width dimension
    """
    n, _, d = x.shape

    x = x.reshape(n, num_patches_h, num_patches_w, d)
    x_masked, len_keep_h = _random_masking_1d(
        x, mask_ratio_h, num_patches_h, num_patches_w
    )
    x_masked = x_masked.transpose(1, 2)
    x_masked, len_keep_w = _random_masking_1d(
        x_masked, mask_ratio_w, num_patches_w, len_keep_h
    )
    x_masked = x_masked.transpose(1, 2)
    x_masked = x_masked.reshape(n, len_keep_h * len_keep_w, d)

    return x_masked


def _random_masking_1d(
    x: Tensor,
    mask_ratio: float,
    num_patches_h: int,
    num_patches_w: int,
) -> Tuple[Tensor, int]:
    # x shape : bsz x h x w x embed_dim
    n, _, _, d = x.shape
    len_keep = int(num_patches_h * (1 - mask_ratio))
    noise = torch.rand(n, num_patches_h, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_patches_w, d)
    x_masked = torch.gather(x, dim=1, index=index)
    return x_masked, len_keep
