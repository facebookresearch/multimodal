# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import Tensor


def get_3d_sin_cos_embeddings(
    embed_dim: int, temporal_size: int, spatial_size: Tuple[int, int]
) -> Tensor:
    """
    3d position sin cos embeddings. This implementation has been adapted from internal
    FAIR implementation: https://fburl.com/code/9dojefjm.
    Args:
        embed_dim (int): embedding dimension of the position embedding
        temporal_size (int): temporal input dimensions of the grid
        spatial_size (Tuple[int, int]): spatial input dimensions of the grid
    return:
        embed (Tensor[int]): [1+temporal_size*spatial_size[0]*spatial_size[1], embed_dim] (w/ cls_token)
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    h, w = spatial_size
    pos_h = torch.arange(h)
    pos_w = torch.arange(w)
    pos_grid = torch.meshgrid(pos_w, pos_h, indexing="xy")
    pos_grid = torch.stack(pos_grid, dim=0).float()

    pos_grid = pos_grid.reshape([2, 1, h, w])

    embed_w = get_1d_sin_cos_embeddings(embed_dim_spatial // 2, pos_grid[0].flatten())
    embed_h = get_1d_sin_cos_embeddings(embed_dim_spatial // 2, pos_grid[1].flatten())
    # h*w x embed_dim_spatial
    embed_spatial = torch.cat([embed_w, embed_h], dim=1)

    # temporal
    pos_t = torch.arange(temporal_size)
    embed_temporal = get_1d_sin_cos_embeddings(embed_dim_temporal, pos_t.flatten())

    # concate: [T, H, W] order
    embed_temporal = embed_temporal[:, None, :]
    embed_temporal = torch.repeat_interleave(
        embed_temporal, h * w, dim=1
    )  # [T, H*W, D // 4]
    embed_spatial = embed_spatial[None, :, :]
    embed_spatial = torch.repeat_interleave(
        embed_spatial, temporal_size, dim=0
    )  # [T, H*W, D // 4 * 3]

    embed = torch.cat([embed_temporal, embed_spatial], dim=-1)
    embed = embed.reshape([-1, embed_dim])  # [T*H*W, D]

    # Add pos embed for cls token
    embed = torch.cat([torch.zeros(1, embed_dim), embed], dim=0)
    embed = embed.unsqueeze(0)
    return embed


def get_2d_sin_cos_embeddings(embed_dim: int, input_size: Tuple[int, int]) -> Tensor:
    """
    2d position sin cos embeddings.
    Args:
        embed_dim (int): embedding dimension of the position embedding
        input_size (Tuple[int, int]): input dimensions of the grid
    """

    # dim gets halved twice, once for h and w axis and once for sin and cos
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    h, w = input_size
    pos_h = torch.arange(h)
    pos_w = torch.arange(w)
    pos_grid = torch.meshgrid(pos_w, pos_h, indexing="xy")
    embed_w = get_1d_sin_cos_embeddings(embed_dim // 2, pos_grid[0].flatten())
    embed_h = get_1d_sin_cos_embeddings(embed_dim // 2, pos_grid[1].flatten())
    # h*w x embed_dim
    embed = torch.cat([embed_w, embed_h], dim=1)
    # Add pos embed for cls token
    embed = torch.cat([torch.zeros(1, embed_dim), embed], dim=0)
    embed = embed.unsqueeze(0)
    return embed


def get_1d_sin_cos_embeddings(embed_dim: int, positions: Tensor) -> Tensor:
    """
    1d position sin cos embeddings.
    Args:
        embed_dim (int): embedding dimension of the position embedding
        positions (Tensor): 1d tensor with the position ids
    """
    omega = 1 / 10000 ** (
        torch.arange(embed_dim // 2, dtype=torch.float) / (embed_dim / 2.0)
    )
    out = torch.einsum("i,j->ij", positions, omega)
    sin_embed = torch.sin(out)
    cos_embed = torch.cos(out)
    embed = torch.cat([sin_embed, cos_embed], dim=1)
    return embed
