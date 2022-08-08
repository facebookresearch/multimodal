# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchmultimodal.models.video_vqvae import (
    preprocess_int_conv_params,
    VideoDecoder,
    VideoEncoder,
)

from torchmultimodal.models.vqvae import VQVAE

MUGEN_PRETRAINED_MAPPING = {
    "mugen_L32": "https://download.pytorch.org/models/multimodal/mugen/mugen_video_vqvae_L32.pt",
    "mugen_L16": "https://download.pytorch.org/models/multimodal/mugen/mugen_video_vqvae_L16.pt",
    "mugen_L8": "https://download.pytorch.org/models/multimodal/mugen/mugen_video_vqvae_L8.pt",
}


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
    pretrained_model_key: Optional[str] = "mugen_L32",
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
    encoder_kernel_sizes_fixed = preprocess_int_conv_params(
        encoder_in_channel_dims, encoder_kernel_size
    )
    decoder_kernel_sizes_fixed = preprocess_int_conv_params(
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
    model = VQVAE(encoder, decoder, num_embeddings, embedding_dim)

    if pretrained_model_key:
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                MUGEN_PRETRAINED_MAPPING[pretrained_model_key]
            )
        )

    return model
