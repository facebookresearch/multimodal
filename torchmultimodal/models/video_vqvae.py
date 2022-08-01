# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast, List, Optional, Tuple

from torch import nn, Tensor

from torchmultimodal.models.vqvae import VQVAE
from torchmultimodal.modules.layers.attention import AxialAttentionBlock
from torchmultimodal.modules.layers.conv import SamePadConv3d, SamePadConvTranspose3d
from torchmultimodal.utils.assertion import assert_equal_lengths
from torchmultimodal.utils.common import to_tuple_tuple


def video_vqvae(
    in_channel_dim: int = 3,
    encoder_hidden_dim: int = 240,
    encoder_kernel_size: int = 3,
    encoder_stride: int = 2,
    encoder_n_layers: int = 1,
    n_res_layers: int = 4,
    attn_hidden_dim: int = 240,
    num_embeddings: int = 1024,
    embedding_dim: int = 256,
    decoder_hidden_dim: int = 240,
    decoder_kernel_size: int = 4,
    decoder_stride: int = 2,
    decoder_n_layers: int = 1,
) -> VQVAE:
    """Generic Video VQVAE builder using default parameters from VideoGPT (Yan et al. 2022).
    Uses hyperparameters from Appendix A Table 8 for BAIR/RoboNet/ViZDoom. Code ref:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channel_dim (int, optional): Size of channel dim in input. Defaults to 3.
        encoder_hidden_dim (int, optional): Size of channel dims in encoder conv layers. Defaults to 240.
        encoder_kernel_size (int, optional): Kernel size for encoder. Defaults to 3.
        encoder_stride (int, optional): Stride for encoder. Defaults to 2.
        encoder_n_layers (int, optional): Number of layers in encoder. Does not include attention stack and pre-codebook conv layer.
            Defaults to 1.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include in encoder and decoder. Defaults to 4.
        attn_hidden_dim (int, optional): Size of hidden dim of ``AttentionResidualBlocks``. Defaults to 240.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``. Defaults to 1024.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``. Defaults to 256.
        decoder_hidden_dim (int, optional): Size of channel dims in decoder conv tranpose layers. Defaults to 240.
        decoder_kernel_size (int, optional): Kernel size for decoder. Defaults to 4.
        decoder_stride (int, optional): Stride for decoder. Defaults to 2.
        decoder_n_layers (int, optional): Number of layers in decoder. Does not include attention stack and
            post-codebook conv transpose layer. Defaults to 1.

    Returns:
        VQVAE: constructed ``VQVAE`` model using ``VideoEncoder``, ``Codebook``, and ``VideoDecoder``
    """

    encoder_in_channel_dims = (in_channel_dim,) + (encoder_hidden_dim,) * max(
        encoder_n_layers - 1, 0
    )
    decoder_out_channel_dims = (decoder_hidden_dim,) * max(decoder_n_layers - 1, 0) + (
        in_channel_dim,
    )

    # Reformat kernel and strides to be tuple of tuple for encoder/decoder constructors
    encoder_kernel_sizes_fixed, encoder_strides_fixed = _preprocess_int_conv_params(
        encoder_in_channel_dims, encoder_kernel_size, encoder_stride
    )
    decoder_kernel_sizes_fixed, decoder_strides_fixed = _preprocess_int_conv_params(
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


def video_vqvae_mugen(
    in_channel_dim: int = 3,
    encoder_hidden_dim: int = 240,
    encoder_kernel_size: int = 3,
    n_res_layers: int = 4,
    attn_hidden_dim: int = 240,
    num_embeddings: int = 2048,
    embedding_dim: int = 256,
    decoder_hidden_dim: int = 240,
    decoder_kernel_size: int = 3,
    pretrained_model_key: Optional[str] = None,
) -> VQVAE:
    """Constructor for MUGEN's Video VQVAE. Expects input video data of shape {8,16,32}x256x256.
    Trained for tokenization of video data and use in video-audio-text retrieval and generation tasks.
    See Hayes et al. 2022 for more details: https://arxiv.org/pdf/2204.08058.pdf
    Code ref:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/video_vqvae/vqvae.py
    https://github.com/mugen-org/MUGEN_baseline/blob/main/generation/experiments/vqvae/VideoVQVAE_L32.sh

    Args:
        in_channel_dim (int, optional): Size of channel dim in input. Defaults to 3.
        encoder_hidden_dim (int, optional): Size of channel dims in encoder conv layers. Defaults to 240.
        encoder_kernel_size (int, optional): Kernel size for encoder. Defaults to 3.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include in encoder and decoder. Defaults to 4.
        attn_hidden_dim (int, optional): Size of hidden dim of ``AttentionResidualBlocks``. Defaults to 240.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``. Defaults to 2048.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``. Defaults to 256.
        decoder_hidden_dim (int, optional): Size of channel dims in decoder conv tranpose layers. Defaults to 240.
        decoder_kernel_size (int, optional): Kernel size for decoder. Defaults to 3.
        pretrained_model_key (str, optional): Load a specified MUGEN VQVAE checkpoint.

    Returns:
        VQVAE: constructed ``VQVAE`` model using ``VideoEncoder``, ``Codebook``, and ``VideoDecoder``
    """
    encoder_strides = ((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (1, 1, 1))
    decoder_strides = ((2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2))
    encoder_n_layers = len(encoder_strides)
    decoder_n_layers = len(decoder_strides)
    encoder_in_channel_dims = (in_channel_dim,) + (encoder_hidden_dim,) * max(
        encoder_n_layers - 1, 0
    )
    decoder_out_channel_dims = (decoder_hidden_dim,) * max(decoder_n_layers - 1, 0) + (
        in_channel_dim,
    )
    encoder_kernel_sizes_fixed = _preprocess_int_conv_params(
        encoder_in_channel_dims, encoder_kernel_size
    )
    decoder_kernel_sizes_fixed = _preprocess_int_conv_params(
        decoder_out_channel_dims, decoder_kernel_size
    )

    encoder = VideoEncoder(
        encoder_in_channel_dims,
        encoder_kernel_sizes_fixed,
        encoder_strides,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
    )
    decoder = VideoDecoder(
        decoder_out_channel_dims,
        decoder_kernel_sizes_fixed,
        decoder_strides,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
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
        in_channel_dims (Tuple[int, ...]): input channel dimension for each layer in conv stack
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): kernel sizes for each layer in conv stack
        strides (Tuple[Tuple[int, int, int], ...]): strides for each layer in conv stack
        output_dim (int): size of hidden dimension of final output
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include. Default is 4.
        attn_hidden_dim (int): size of hidden dimension in attention block. Default is 240.

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

    def forward(self, x: Tensor) -> Tensor:
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
    """Decoder for Video VQVAE. Takes quantized output from codebook and applies a
    ``SamePadConv3d`` layer, a stack of ``AttentionResidualBlocks``, followed by a
    specified number of ``SamePadConvTranspose3d`` layers. The residual
    blocks use Axial Attention to enhance representations of video data without
    significantly increasing computational cost. Follows VideoGPT's implementation:
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        out_channel_dims (Tuple[int, ...]): output channel dimension for each layer in conv stack
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): kernel sizes for each layer in conv stack
        strides (Tuple[Tuple[int, int, int], ...]): strides for each layer in conv stack
        input_dim (int): input channel dimension for first conv layer before attention stack
        n_res_layers (int): number of ``AttentionResidualBlocks`` to include. Default is 4.
        attn_hidden_dim (int): size of hidden dimension in attention block. Default is 240.
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
                          projections Wq, Wk, and Wv in attention. Default is 240.
        n_head (int): number of heads in multihead attention. Must divide into hidden_dim
                      evenly. Default is 2 from VideoGPT.

    Inputs:
        x (Tensor): a [b, c, d1, ..., dn] tensor
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
        return x + self.block(x)


def _preprocess_int_conv_params(
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
