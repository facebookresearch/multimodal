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
from torchmultimodal.utils.common import format_convnet_params

DEFAULT_ENCODER_IN_CHANNEL_DIMS = (3, 240, 240, 240, 240, 240)
DEFAULT_ENCODER_STRIDES = (
    (2, 2, 2),
    (2, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 1, 1),
)
DEFAULT_DECODER_OUT_CHANNEL_DIMS = (240, 240, 240, 240, 3)
DEFAULT_DECODER_STRIDES = (
    (2, 2, 2),
    (2, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
    (1, 2, 2),
)


def video_vqvae(
    encoder_in_channel_dims: Tuple[int, ...] = DEFAULT_ENCODER_IN_CHANNEL_DIMS,
    encoder_kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]] = 3,
    encoder_strides: Union[
        int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]
    ] = DEFAULT_ENCODER_STRIDES,
    n_res_layers: int = 4,
    attn_hidden_dim: int = 240,
    num_embeddings: int = 2048,
    embedding_dim: int = 256,
    decoder_out_channel_dims: Tuple[int, ...] = DEFAULT_DECODER_OUT_CHANNEL_DIMS,
    decoder_kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]] = 3,
    decoder_strides: Union[
        int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]
    ] = DEFAULT_DECODER_STRIDES,
) -> VQVAE:
    """Construct Video VQVAE with default parameters used in MUGEN (Hayes et al. 2022). Code ref:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/generation/experiments/vqvae/VideoVQVAE_L8.sh

    Args:
        encoder_in_channel_dims (Tuple[int, ...], optional): See ``VideoEncoder``. Defaults to (3, 240, 240, 240, 240, 240).
        encoder_kernel_sizes (Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], optional): See ``VideoEncoder``.
                                                                                       Defaults to (3, 3, 3, 3, 3, 3).
        encoder_strides (Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], optional): See ``VideoEncoder``.
            Defaults to ( (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1), ).
        n_res_layers (int, optional): See ``VideoEncoder``. Used in both encoder and decoder. Defaults to 4.
        attn_hidden_dim (int, optional): See ``VideoEncoder``. Used in both encoder and decoder. Defaults to 240.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``. Defaults to 2048.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``. Defaults to 256.
        decoder_out_channel_dims (Tuple[int, ...], optional): See ``VideoDecoder``. Defaults to (240, 240, 240, 240, 3).
        decoder_kernel_sizes (Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], optional): See ``VideoDecoder``.
            Defaults to 3.
        decoder_strides (Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]], optional): See ``VideoDecoder``.
            Defaults to ( (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), ).

    Returns:
        VQVAE: constructed ``VQVAE`` model using ``VideoEncoder``, ``Codebook``, and ``VideoDecoder``
    """
    encoder = VideoEncoder(
        encoder_in_channel_dims,
        encoder_kernel_sizes,
        encoder_strides,
        n_res_layers,
        attn_hidden_dim,
        embedding_dim,
    )
    decoder = VideoDecoder(
        decoder_out_channel_dims,
        decoder_kernel_sizes,
        decoder_strides,
        n_res_layers,
        attn_hidden_dim,
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
        in_channel_dims (Tuple[int, ...]): input channel dimension for each conv layer
        kernel_sizes (int or Tuple[int, int, int] or Tuple[Tuple[int, int, int], ...]): kernel sizes for each conv layer
        strides (int or Tuple[int, int, int] or Tuple[Tuple[int, int, int], ...]): strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        attn_hidden_dim (int): size of hidden dimension in attention block
        embedding_dim (int): size of hidden dimension of final output
        **kwargs (dict): keyword arguments to be passed into ``SamePadConv3d`` and used by ``nn.Conv3d``

    Raises:
        ValueError: if the lengths of ``in_channel_dims``, ``kernel_sizes``,
                    and ``strides`` are not all equivalent

    Inputs:
        x (Tensor): input video data with shape (b x c x d1 x d2 x d3)
    """

    def __init__(
        self,
        in_channel_dims: Tuple[int, ...],
        kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_res_layers: int,
        attn_hidden_dim: int,
        embedding_dim: int,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        in_channel_dims, kernel_sizes, strides = format_convnet_params(
            in_channel_dims, kernel_sizes, strides, 3
        )

        convolutions: List[nn.Module] = []
        n_conv_layers = len(in_channel_dims)
        for i in range(n_conv_layers):
            in_channel = in_channel_dims[i]
            out_channel = (
                in_channel_dims[i + 1] if i < n_conv_layers - 1 else attn_hidden_dim
            )
            kernel = kernel_sizes[i]
            stride = strides[i]
            convolutions.append(
                SamePadConv3d(
                    in_channel, out_channel, kernel, stride, bias=True, **kwargs
                )
            )
            # Do not apply relu to last conv layer before res stack
            if i < n_conv_layers - 1:
                convolutions.append(nn.ReLU())
        self.convs = nn.Sequential(*convolutions)

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU(),
        )

        self.conv_out = SamePadConv3d(
            attn_hidden_dim, embedding_dim, kernel_size=1, stride=1
        )

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
        out_channel_dims (Tuple[int, ...]): output channel dimension for each conv layer
        kernel_sizes (int or Tuple[int, int, int] or Tuple[Tuple[int, int, int], ...]): kernel sizes for each conv layer
        strides (int or Tuple[int, int, int] or Tuple[Tuple[int, int, int], ...]): strides for each conv layer
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include
        attn_hidden_dim (int): size of hidden dimension in attention block
        embedding_dim (int): size of hidden dimension of input
        **kwargs (dict): keyword arguments to be passed into ``SamePadConvTranspose3d``
                         and used by ``nn.ConvTranspose3d``

    Raises:
        ValueError: if the lengths of ``out_channel_dims``, ``kernel_sizes``,
                    and ``strides`` are not all equivalent
        ValueError: if input Tensor channel dim does not match ``embedding_dim``

    Inputs:
        x (Tensor): input tokenized data with shape (b x c x d1 x d2 x d3)
    """

    def __init__(
        self,
        out_channel_dims: Tuple[int, ...],
        kernel_sizes: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_res_layers: int,
        attn_hidden_dim: int,
        embedding_dim: int,
        **kwargs: Dict[str, Any],
    ):
        super().__init__()
        out_channel_dims, kernel_sizes, strides = format_convnet_params(
            out_channel_dims, kernel_sizes, strides, 3
        )

        self.conv_in = SamePadConv3d(
            embedding_dim, attn_hidden_dim, kernel_size=1, stride=1
        )

        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)],
            nn.BatchNorm3d(attn_hidden_dim),
            nn.ReLU(),
        )

        transpose_convolutions: List[nn.Module] = []
        n_conv_layers = len(out_channel_dims)
        for i in range(n_conv_layers):
            in_channel = out_channel_dims[i - 1] if i > 0 else attn_hidden_dim
            out_channel = out_channel_dims[i]
            kernel = kernel_sizes[i]
            stride = strides[i]
            transpose_convolutions.append(
                SamePadConvTranspose3d(
                    in_channel, out_channel, kernel, stride, bias=True, **kwargs
                )
            )
            # Do not apply relu to output convt layer
            if i < n_conv_layers - 1:
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
