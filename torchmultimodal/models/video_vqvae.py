# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Union

from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.modules.layers.attention import AxialAttentionBlock

from torchmultimodal.modules.layers.conv import SamePadConv3d, SamePadConvTranspose3d


class VideoEncoder(nn.Module):
    """Encoder for Video VQVAE. Stacks specified number of ``SamePadConv3d`` layers
    followed by a stack of ``AttentionResidualBlocks``. The residual blocks use Axial
    Attention to enhance representations of video data without significantly
    increasing computational cost. Follows VideoGPT's implementation:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (List[int]): list of input channel dimension for each conv layer
        out_channels (List[int]): list of output channel dimension for each conv layer
        kernel_sizes (List[int or Tuple[int]]): list of kernel sizes for each conv layer
        strides (List[int or Tuple[int]]): list of strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        **kwargs (dict): keyword arguments to be passed into ``SamePadConv3d`` and used by ``nn.Conv3d``

    Raises:
        ValueError: if the lengths of ``in_channels``, ``out_channels``, ``kernel_sizes``,
                    and ``strides`` are not all equivalent
        ValueError: if ``in_channels`` is not identical to ``out_channels`` offset by one

    Inputs:
        x (Tensor): input video data with shape (b x c x d1 x d2 x d3)
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int, int]]],
        strides: List[Union[int, Tuple[int, int, int]]],
        n_res_layers: int,
        **kwargs: Dict[str, Any]
    ):
        super().__init__()
        if not (
            len(in_channels) == len(out_channels) == len(kernel_sizes) == len(strides)
        ):
            raise ValueError(
                "in_channels, out_channels, kernel_sizes, strides should all have the same length"
            )
        if in_channels[1:] != out_channels[:-1]:
            raise ValueError("out_channels should match in_channels offset by one")

        self.convs = nn.ModuleList(
            [
                SamePadConv3d(i, o, k, s, bias=True, **kwargs)
                for i, o, k, s in zip(in_channels, out_channels, kernel_sizes, strides)
            ]
        )
        attn_hidden_dim = out_channels[-1]
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for conv in self.convs[:-1]:
            h = F.relu(conv(h))
        # Do not apply relu to last conv layer before res stack
        h = self.convs[-1](h)
        h = self.res_stack(h)
        return h


class VideoDecoder(nn.Module):
    """Decoder for Video VQVAE. Takes quantized output from codebook and applies stack of
    ``AttentionResidualBlocks``, followed by specified number of ``SamePadConvTranspose3d``
    layers. The residual blocks use Axial Attention to enhance representations of video
    data without significantly increasing computational cost. Follows VideoGPT's implementation:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (List[int]): list of input channel dimension for each conv layer
        out_channels (List[int]): list of output channel dimension for each conv layer
        kernel_sizes (List[int or Tuple[int]]): list of kernel sizes for each conv layer
        strides (List[int or Tuple[int]]): list of strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        **kwargs (dict): keyword arguments to be passed into ``SamePadConvTranspose3d``
                         and used by ``nn.ConvTranspose3d``

    Raises:
        ValueError: if the lengths of ``in_channels``, ``out_channels``, ``kernel_sizes``,
                    and ``strides`` are not all equivalent
        ValueError: if ``in_channels`` is not identical to ``out_channels`` offset by one

    Inputs:
        x (Tensor): input tokenized data with shape (b x c x d1 x d2 x d3)
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_sizes: List[Union[int, Tuple[int, int, int]]],
        strides: List[Union[int, Tuple[int, int, int]]],
        n_res_layers: int,
        **kwargs: Dict[str, Any]
    ):
        super().__init__()
        if not (
            len(in_channels) == len(out_channels) == len(kernel_sizes) == len(strides)
        ):
            raise ValueError(
                "in_channels, out_channels, kernel_sizes, strides should all have the same length"
            )
        if in_channels[1:] != out_channels[:-1]:
            raise ValueError("out_channels should match in_channels offset by one")

        attn_hidden_dim = in_channels[0]
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU()
        )
        self.convts = nn.ModuleList(
            [
                SamePadConvTranspose3d(i, o, k, s, bias=True, **kwargs)
                for i, o, k, s in zip(in_channels, out_channels, kernel_sizes, strides)
            ]
        )

    def forward(self, x: Tensor):
        h = self.res_stack(x)
        for convt in self.convts[:-1]:
            h = F.relu(convt(h))
        # Do not apply relu to output convt layer
        h = self.convts[-1](h)
        return h


class AttentionResidualBlock(nn.Module):
    """Residual block with axial attention as designed by VideoGPT (Yan et al. 2021)
    and used in MUGEN (Hayes et al. 2022). Follows implementation by VideoGPT:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        hidden_dim (int): size of channel dim of input. Also determines dim of linear
                          projections Wq, Wk, and Wv in attention
        n_head (int): number of heads in multihead attention. Must divide into hidden_dim
                      evenly. Default is 2 from VideoGPT.

    Inputs:
        x (Tensor): a [b, c, d1, ..., dn] tensor
    """

    def __init__(self, hidden_dim: int, n_head: int = 2) -> None:
        super().__init__()
        # To avoid hidden dim becoming 0 in middle layers
        if hidden_dim < 2:
            raise ValueError("hidden dim must be at least 2")

        self.block = nn.Sequential(
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            SamePadConv3d(hidden_dim, hidden_dim // 2, 3, bias=False),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            SamePadConv3d(hidden_dim // 2, hidden_dim, 1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            AxialAttentionBlock(3, hidden_dim, n_head),  # Video has 3 dims
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)
