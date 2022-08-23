# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

from torch import nn
from torchmultimodal.models.gpt import (
    MultimodalGPT,
    MultimodalTransformerDecoder,
    RightShift,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torchmultimodal.models.video_vqvae import VideoDecoder, VideoEncoder
from torchmultimodal.models.vqvae import VQVAE

from torchmultimodal.modules.layers.attention import SelfAttention
from torchmultimodal.modules.layers.position_embedding import (
    BroadcastedPositionEmbedding,
)


def video_gpt(
    input_shape: Tuple[int, int, int] = (16, 64, 64),
    latent_shape: Tuple[int, int, int] = (8, 32, 32),
    d_model: int = 576,  # https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/gpt.py#L177
    n_head: int = 4,
    dropout: float = 0.2,
    attn_dropout: float = 0.3,
    num_decoder_layers: int = 16,
) -> MultimodalGPT:
    """VideoGPT model

    For the momemnt this is a prototype implementation which may not fully commit to the specifications
    described in the paper `"VideoGPT: Video Generation using VQ-VAE and Transformers
    "<https://arxiv.org/pdf/2104.10157.pdf>`_.
    Source of parameters:
        * Page 13 Table A.1 Column "BAIR / RoboNet / ViZDoom"
        * Page 13 Table A.2 Column "BAIR / RoboNet"

    Args:
        input_shape (Tuple[int, int, int]): Shape of the input video data ``(time_seq_len, resolution, resolution)``.
            Defaults to ``(16, 64, 64)``.
        latent_shape (Tuple[int, int, int]): Shape of the encoded video data. This should be consistent with the actual
            latent shape inferred by the video encoder, see :class:`~torchmultimodal.models.video_vqvae.VideoEncoder`.
            Defaults to ``(8, 32, 32)``.
        d_model (int): Dimension of the underlying transformer decoder.
            See :py:class:`torchmultimodal.models.gpt.TransformerDecoderLayer`. Defaults to ``576``.
        n_head (int): Number of attention heads used by the transformer decoder. Defaults to ``4``.
        dropout (float): Dropout probability used by the projection layer of the transformer decoder.
            Defaults to ``0.2``.
        attn_dropout (float): Dropout probability used by the attention layer of the transformer decoder.
            Defaults to ``0.3``.
        num_decoder_layers (int): Number of transformer decoder layers. Defaults to ``16``.

    Returns:
        An instance of :class:`~torchmultimodal.models.gpt.MultimodalGPT`.
    """
    # constructs in and out tokenizers
    in_tokenizer = video_vqvae()
    out_tokenizer = video_vqvae()
    num_in_tokens = in_tokenizer.num_embeddings  # codebook size
    num_out_tokens = out_tokenizer.num_embeddings

    # derived parameters
    vqvae_latent_shape = in_tokenizer.latent_shape(input_shape)
    if latent_shape != vqvae_latent_shape:
        raise ValueError(
            f"Latent shape required: {latent_shape} does not match that of VQVAE: {vqvae_latent_shape}"
        )

    # constructs projection layers
    in_projection = nn.Linear(in_tokenizer.embedding_dim, d_model, bias=False)
    in_projection.weight.data.normal_(std=0.02)
    out_projection = nn.Linear(out_tokenizer.embedding_dim, d_model, bias=False)
    out_projection.weight.data.normal_(std=0.02)

    # constructs multimodal decoder
    in_pos_emb = BroadcastedPositionEmbedding(latent_shape, d_model)
    out_pos_emb = BroadcastedPositionEmbedding(latent_shape, d_model)
    attention_layer = SelfAttention(attn_dropout=attn_dropout)
    decoder_layer = TransformerDecoderLayer(
        d_model, n_head, dropout, attn_module=attention_layer
    )
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    right_shift = RightShift(d_model)
    mm_decoder = MultimodalTransformerDecoder(
        in_pos_emb, out_pos_emb, decoder, right_shift
    )

    return MultimodalGPT(
        d_model=d_model,
        num_in_tokens=num_in_tokens,
        num_out_tokens=num_out_tokens,
        latent_shape=latent_shape,
        in_tokenizer=in_tokenizer,
        out_tokenizer=out_tokenizer,
        mm_decoder=mm_decoder,
        in_projection=in_projection,
        out_projection=out_projection,
    )


def video_vqvae(
    in_channel_dim: int = 3,
    encoder_hidden_dim: int = 240,
    n_res_layers: int = 4,
    attn_hidden_dim: int = 240,
    num_embeddings: int = 1024,
    embedding_dim: int = 256,
    decoder_hidden_dim: int = 240,
) -> VQVAE:
    """Video VQVAE builder for VideoGPT

    Args:
        in_channel_dim (int, optional): Size of channel dim in input. Defaults to ``3``.
        encoder_hidden_dim (int, optional): Size of channel dims in encoder conv layers. Defaults to ``240``.
        n_res_layers (int, optional): Number of :class:`~torchmultimodal.models.video_vqvae.AttentionResidualBlocks`
            to include in encoder and decoder. Defaults to ``4``.
        attn_hidden_dim (int, optional): Size of hidden dim of ``AttentionResidualBlocks``. Defaults to ``240``.
        num_embeddings (int, optional): Number of embedding vectors used in ``Codebook``. Defaults to ``1024``.
        embedding_dim (int, optional): Dimensionality of embedding vectors in ``Codebook``. Defaults to ``256``.
        decoder_hidden_dim (int, optional): Size of channel dims in decoder conv tranpose layers. Defaults to ``240``.

    Returns:
        An instance of :class:`~torchmultimodal.models.vqvae.VQVAE` constructed with:
            * :class:`~torchmultimodal.model.video_vqvae.VideoEncoder`
            * :class:`~torchmultimodal.model.video_vqvae.VideoDecoder`
    """

    # this includes both the upsampling conv filter size and encoder filter size
    # ideally we should code them as two separate parameters in VideoEncoder
    encoder_kernel_sizes = ((4, 4, 4), (3, 3, 3))
    # stride (1, 1, 1) is that of the encoder filter conv layer
    encoder_strides = ((2, 2, 2), (1, 1, 1))
    encoder_n_layers = len(encoder_strides)
    decoder_kernel_sizes = ((4, 4, 4),)
    decoder_strides = ((2, 2, 2),)
    decoder_n_layers = len(decoder_strides)

    encoder_in_channel_dims = (in_channel_dim,) + (encoder_hidden_dim,) * max(
        encoder_n_layers - 1, 0
    )
    decoder_out_channel_dims = (decoder_hidden_dim,) * max(decoder_n_layers - 1, 0) + (
        in_channel_dim,
    )

    encoder = VideoEncoder(
        encoder_in_channel_dims,
        encoder_kernel_sizes,
        encoder_strides,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
    )
    decoder = VideoDecoder(
        decoder_out_channel_dims,
        decoder_kernel_sizes,
        decoder_strides,
        embedding_dim,
        n_res_layers,
        attn_hidden_dim,
    )

    return VQVAE(encoder, decoder, num_embeddings, embedding_dim)
