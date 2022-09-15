# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List, Optional, Tuple, Union

from torch import nn, Size, Tensor

from torchmultimodal.models.vqvae import VQVAE
from torchmultimodal.modules.layers.attention import AxialAttentionBlock
from torchmultimodal.modules.layers.conv import SamePadConv3d, SamePadConvTranspose3d
from torchmultimodal.utils.assertion import assert_equal_lengths
from torchmultimodal.utils.common import to_tuple_tuple


def video_vqvae(
    in_channel_dim: int,
    encoder_hidden_dim: int,
    encoder_kernel_size: int,
    encoder_stride: int,
    encoder_n_layers: int,
    n_res_layers: int,
    attn_hidden_dim: int,
    num_embeddings: int,
    embedding_dim: int,
    decoder_hidden_dim: int,
    decoder_kernel_size: int,
    decoder_stride: int,
    decoder_n_layers: int,
) -> VQVAE:
    """Generic Video VQVAE builder

    Args:
        in_channel_dim (int, optional): Size of channel dim in input.
        encoder_hidden_dim (int, optional): Size of channel dims in encoder conv layers.
        encoder_kernel_size (int, optional): Kernel size for encoder.
        encoder_stride (int, optional): Stride for encoder.
        encoder_n_layers (int, optional): Number of layers in encoder. Does not include attention stack
            and pre-codebook conv layer.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include in encoder and decoder.
        attn_hidden_dim (int, optional): Size of hidden dim of ``AttentionResidualBlocks``.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``.
        decoder_hidden_dim (int, optional): Size of channel dims in decoder conv transpose layers.
        decoder_kernel_size (int, optional): Kernel size for decoder.
        decoder_stride (int, optional): Stride for decoder.
        decoder_n_layers (int, optional): Number of layers in decoder. Does not include attention stack and
            post-codebook conv transpose layer.

    Returns:
        An instance of :class:`~torchmultimodal.models.vqvae.VQVAE` initialized with ``VideoEncoder``,
            ``Codebook`` and ``VideoDecoder``
    """

    encoder_in_channel_dims = (in_channel_dim,) + (encoder_hidden_dim,) * max(
        encoder_n_layers - 1, 0
    )
    decoder_out_channel_dims = (decoder_hidden_dim,) * max(decoder_n_layers - 1, 0) + (
        in_channel_dim,
    )

    # Reformat kernel and strides to be tuple of tuple for encoder/decoder constructors
    encoder_kernel_sizes_fixed, encoder_strides_fixed = preprocess_int_conv_params(
        encoder_in_channel_dims, encoder_kernel_size, encoder_stride
    )
    decoder_kernel_sizes_fixed, decoder_strides_fixed = preprocess_int_conv_params(
        decoder_out_channel_dims, decoder_kernel_size, decoder_stride
    )

    encoder = VideoEncoder(
        encoder_in_channel_dims,
        encoder_kernel_sizes_fixed,
        encoder_strides_fixed,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
    )
    decoder = VideoDecoder(
        decoder_out_channel_dims,
        decoder_kernel_sizes_fixed,
        decoder_strides_fixed,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
    )

    return VQVAE(encoder, decoder, num_embeddings, embedding_dim)


class VideoEncoder(nn.Module):
    """Encoder for Video VQVAE.

    Stacks specified number of ``SamePadConv3d`` layers
    followed by a stack of ``AttentionResidualBlocks`` and a final ``SamePadConv3d``
    layer before the codebook. The residual blocks use Axial Attention to enhance
    representations of video data without significantly increasing computational
    cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channel_dims (Tuple[int, ...]): Input channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack.
        output_dim (int): Size of hidden dimension of final output.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int, optional): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConv3d`` and used by ``nn.Conv3d``.

    Raises:
        ValueError: If the lengths of ``in_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(
        self,
        in_channel_dims: Tuple[int, ...],
        kernel_sizes: Tuple[Tuple[int, int, int], ...],
        strides: Tuple[Tuple[int, int, int], ...],
        output_dim: int,
        n_res_layers: int = 4,
        attn_hidden_dim: int = 240,
        **kwargs: Any,
    ):
        super().__init__()

        assert_equal_lengths(
            in_channel_dims,
            kernel_sizes,
            strides,
            msg="in_channel_dims, kernel_sizes, and strides must be same length.",
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
            attn_hidden_dim, output_dim, kernel_size=1, stride=1
        )

    def get_latent_shape(self, input_shape: Union[Tuple, Size]) -> Tuple:
        """Return shape of encoder output based on number of downsampling conv layers"""
        latent_shape = list(input_shape)
        for layer in self.convs:  # ignore conv_out since it has a stride of 1
            if isinstance(layer, SamePadConv3d):
                # SamePadConv should downsample input shape by factor of stride
                latent_shape = [
                    latent_shape[dim] // layer.conv.stride[dim]
                    for dim in range(len(input_shape))
                ]

        return tuple(latent_shape)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input video data with shape ``(b, c, d1, d2, d3)``.
        """
        in_channel = x.shape[1]
        if in_channel != self.convs[0].conv.in_channels:
            raise ValueError(
                f"expected input channel dim to be {self.convs[0].conv.in_channels}, but got {in_channel}"
            )
        h = self.convs(x)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return h


class VideoDecoder(nn.Module):
    """Decoder for Video VQVAE.

    Takes quantized output from codebook and applies a ``SamePadConv3d`` layer, a stack of
    ``AttentionResidualBlocks``, followed by a specified number of ``SamePadConvTranspose3d``
    layers. The residual blocks use Axial Attention to enhance representations of video data
    without significantly increasing computational cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        out_channel_dims (Tuple[int, ...]): Output channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack
        input_dim (int): Input channel dimension for first conv layer before attention stack
        n_res_layers (int): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConvTranspose3d`` and used by
            ``nn.ConvTranspose3d``.

    Raises:
        ValueError: If the lengths of ``out_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(
        self,
        out_channel_dims: Tuple[int, ...],
        kernel_sizes: Tuple[Tuple[int, int, int], ...],
        strides: Tuple[Tuple[int, int, int], ...],
        input_dim: int,
        n_res_layers: int = 4,
        attn_hidden_dim: int = 240,
        **kwargs: Any,
    ):
        super().__init__()

        assert_equal_lengths(
            out_channel_dims,
            kernel_sizes,
            strides,
            msg="out_channel_dims, kernel_sizes, and strides must be same length.",
        )

        self.conv_in = SamePadConv3d(
            input_dim, attn_hidden_dim, kernel_size=1, stride=1
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
        """
        Args:
            x (Tensor): Input quantized embeddings with shape ``(b, emb_dim, d1, d2, d3)``.
        """
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
    """Residual block with axial attention.

    Implements the component as proposed in `"VideoGPT: Video Generation using VQ-VAE and
    Transformers (Yan et al. 2022)"<https://arxiv.org/pdf/2104.10157.pdf>`_.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        hidden_dim (int, optional): Size of channel dim of input. Default is ``240``.
        n_head (int, optional): Number of heads in multihead attention. Must divide into hidden_dim evenly.
            Default is ``2``.

    Raises:
        ValueError: If ``hidden_dim`` is less than ``2``.
    """

    def __init__(self, hidden_dim: int = 240, n_head: int = 2) -> None:
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
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        return x + self.block(x)


def preprocess_int_conv_params(
    channel_dims: Tuple[int, ...],
    kernel_sizes: Optional[int] = None,
    strides: Optional[int] = None,
) -> Tuple:
    """Reformats conv params from int to tuple of tuple and assigns correct type"""
    if kernel_sizes is None and strides is None:
        raise ValueError("must specify at least one of kernel_sizes or strides")
    kernel_sizes_fixed = None
    strides_fixed = None
    n_conv_layers = len(channel_dims)
    if kernel_sizes:
        kernel_sizes_fixed = to_tuple_tuple(
            kernel_sizes, dim_tuple=3, num_tuple=n_conv_layers
        )
        kernel_sizes_fixed = cast(Tuple[Tuple[int, int, int], ...], kernel_sizes_fixed)
    if strides:
        strides_fixed = to_tuple_tuple(strides, dim_tuple=3, num_tuple=n_conv_layers)
        strides_fixed = cast(Tuple[Tuple[int, int, int], ...], strides_fixed)

    if kernel_sizes_fixed and strides_fixed:
        return kernel_sizes_fixed, strides_fixed
    elif kernel_sizes_fixed:
        return kernel_sizes_fixed
    else:
        return strides_fixed
