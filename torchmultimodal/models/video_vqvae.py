# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Union

from torch import nn, Tensor

from torchmultimodal.models.vqvae import VQVAE
from torchmultimodal.modules.layers.attention import AxialAttentionBlock
from torchmultimodal.modules.layers.conv import SamePadConv3d, SamePadConvTranspose3d


def video_vqvae(
    encoder_in_channels: Tuple[int, ...] = (3, 240, 240, 240, 240, 240),
    encoder_out_channels: Tuple[int, ...] = (240, 240, 240, 240, 240, 240),
    encoder_kernel_sizes: Tuple[Union[int, Tuple[int, int, int]], ...] = (
        3,
        3,
        3,
        3,
        3,
        3,
    ),
    encoder_strides: Tuple[Union[int, Tuple[int, int, int]], ...] = (
        (2, 2, 2),
        (2, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
        (1, 1, 1),
    ),
    encoder_res_layers: int = 4,
    num_embeddings: int = 2048,
    embedding_dim: int = 256,
    decoder_in_channels: Tuple[int, ...] = (240, 240, 240, 240, 240),
    decoder_out_channels: Tuple[int, ...] = (240, 240, 240, 240, 3),
    decoder_kernel_sizes: Tuple[Union[int, Tuple[int, int, int]], ...] = (
        3,
        3,
        3,
        3,
        3,
    ),
    decoder_strides: Tuple[Union[int, Tuple[int, int, int]], ...] = (
        (2, 2, 2),
        (2, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
        (1, 2, 2),
    ),
    decoder_res_layers: int = 4,
) -> VQVAE:
    """Construct Video VQVAE with default parameters used in MUGEN (Hayes et al. 2022). Code ref:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/generation/experiments/vqvae/VideoVQVAE_L8.sh

    Args:
        encoder_in_channels (Tuple[int, ...], optional): See ``VideoEncoder``. Defaults to (3, 240, 240, 240, 240, 240).
        encoder_out_channels (Tuple[int, ...], optional): See ``VideoEncoder``. Defaults to (240, 240, 240, 240, 240, 240).
        encoder_kernel_sizes (Tuple[Union[int, Tuple[int, int, int]], ...], optional): See ``VideoEncoder``.
                                                                                       Defaults to (3, 3, 3, 3, 3, 3).
        encoder_strides (Tuple[Union[int, Tuple[int, int, int]], ...], optional): See ``VideoEncoder``.
            Defaults to ( (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1), ).
        encoder_res_layers (int, optional): See ``VideoEncoder``. Defaults to 4.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``. Defaults to 2048.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``. Defaults to 256.
        decoder_in_channels (Tuple[int, ...], optional): See ``VideoDecoder``. Defaults to (240, 240, 240, 240, 240).
        decoder_out_channels (Tuple[int, ...], optional): See ``VideoDecoder``. Defaults to (240, 240, 240, 240, 3).
        decoder_kernel_sizes (Tuple[Union[int, Tuple[int, int, int]], ...], optional): See ``VideoDecoder``.
            Defaults to (3, 3, 3, 3, 3).
        decoder_strides (Tuple[Union[int, Tuple[int, int, int]], ...], optional): See ``VideoDecoder``.
            Defaults to ( (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), ).
        decoder_res_layers (int, optional): See ``VideoDecoder``. Defaults to 4.

    Returns:
        VQVAE: constructed ``VQVAE`` model using ``VideoEncoder``, ``Codebook``, and ``VideoDecoder``
    """
    encoder = VideoEncoder(
        encoder_in_channels,
        encoder_out_channels,
        encoder_kernel_sizes,
        encoder_strides,
        encoder_res_layers,
        embedding_dim,
    )
    decoder = VideoDecoder(
        decoder_in_channels,
        decoder_out_channels,
        decoder_kernel_sizes,
        decoder_strides,
        decoder_res_layers,
        embedding_dim,
    )
    return VQVAE(encoder, decoder, num_embeddings, embedding_dim)


class VideoEncoder(nn.Module):
    """Encoder for Video VQVAE. Stacks specified number of ``SamePadConv3d`` layers
    followed by a stack of ``AttentionResidualBlocks`` and a final ``SamePadConv3d``
    layer before the codebook. The residual blocks use Axial Attention to enhance
    representations of video data without significantly increasing computational
    cost. Follows VideoGPT's implementation:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (Tuple[int]): tuple of input channel dimension for each conv layer
        out_channels (Tuple[int]): tuple of output channel dimension for each conv layer
        kernel_sizes (Tuple[int or Tuple[int]]): tuple of kernel sizes for each conv layer
        strides (Tuple[int or Tuple[int]]): tuple of strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        embedding_dim (int): size of hidden dimension of final output
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
        in_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        kernel_sizes: Tuple[Union[int, Tuple[int, int, int]], ...],
        strides: Tuple[Union[int, Tuple[int, int, int]], ...],
        n_res_layers: int,
        embedding_dim: int,
        **kwargs: Dict[str, Any],
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

        convolutions: List[nn.Module] = []
        for idx, (i, o, k, s) in enumerate(
            zip(in_channels, out_channels, kernel_sizes, strides)
        ):
            convolutions.append(SamePadConv3d(i, o, k, s, bias=True, **kwargs))
            # Do not apply relu to last conv layer before res stack
            if idx < len(strides) - 1:
                convolutions.append(nn.ReLU())
        self.convs = nn.Sequential(*convolutions)

        attn_hidden_dim = out_channels[-1]
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU(),
        )

        self.conv_out = SamePadConv3d(attn_hidden_dim, embedding_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.convs(x)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return h


class VideoDecoder(nn.Module):
    """Decoder for Video VQVAE. Takes quantized output from codebook and applies a
    ``SamePadConv3d`` layer, a stack of ``AttentionResidualBlocks``, followed by a
    specified number of ``SamePadConvTranspose3d`` layers. The residual
    blocks use Axial Attention to enhance representations of video data without
    significantly increasing computational cost. Follows VideoGPT's implementation:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (Tuple[int]): tuple of input channel dimension for each conv layer
        out_channels (Tuple[int]): tuple of output channel dimension for each conv layer
        kernel_sizes (Tuple[int or Tuple[int]]): tuple of kernel sizes for each conv layer
        strides (Tuple[int or Tuple[int]]): tuple of strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        embedding_dim (int): size of hidden dimension of input
        **kwargs (dict): keyword arguments to be passed into ``SamePadConvTranspose3d``
                         and used by ``nn.ConvTranspose3d``

    Raises:
        ValueError: if the lengths of ``in_channels``, ``out_channels``, ``kernel_sizes``,
                    and ``strides`` are not all equivalent
        ValueError: if ``in_channels`` is not identical to ``out_channels`` offset by one
        ValueError: if input Tensor channel dim does not match ``embedding_dim``

    Inputs:
        x (Tensor): input tokenized data with shape (b x c x d1 x d2 x d3)
    """

    def __init__(
        self,
        in_channels: Tuple[int, ...],
        out_channels: Tuple[int, ...],
        kernel_sizes: Tuple[Union[int, Tuple[int, int, int]], ...],
        strides: Tuple[Union[int, Tuple[int, int, int]], ...],
        n_res_layers: int,
        embedding_dim: int,
        **kwargs: Dict[str, Any],
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
        self.conv_in = SamePadConv3d(embedding_dim, attn_hidden_dim, 1)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU(),
        )
        transpose_convolutions: List[nn.Module] = []
        for idx, (i, o, k, s) in enumerate(
            zip(in_channels, out_channels, kernel_sizes, strides)
        ):
            transpose_convolutions.append(
                SamePadConvTranspose3d(i, o, k, s, bias=True, **kwargs)
            )
            # Do not apply relu to output convt layer
            if idx < len(strides) - 1:
                transpose_convolutions.append(nn.ReLU())
        self.convts = nn.Sequential(*transpose_convolutions)

    def forward(self, x: Tensor) -> Tensor:
        in_channel = x.shape[1]
        if in_channel != self.conv_in.conv.in_channels:
            raise ValueError(
                f"expected input channel dim to be {self.conv_in.conv.in_channels}, but got {in_channel}"
            )
        h = self.conv_in(x)
        h = self.res_stack(h)
        h = self.convts(h)
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
