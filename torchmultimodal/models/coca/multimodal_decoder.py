# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import TransformerDecoder
from torchmultimodal.utils.attention import get_causal_attention_mask


class CoCaMultimodalDecoder(nn.Module):
    """
    Multimodal decoder for CoCa model.
    Uses a transformer decoder with causal mask for text embeddings
    that cross-attends to image embeddings, followed by output projection.
    Based on the implementation in open_clip: https://tinyurl.com/mn35vdmd

    Args:
        input_seq_len (int): Number of text positions (used to construct
            causal mask)
        text_embedding_dim (int): Dimension of text embeddings
            inside transformer decoder.
        n_layer (int): Number of transformer layers
        n_head (int): Number of heads in multi-head attention
        dim_feedforward (int): Dimension of FFN in transformer decoder
        dropout (float): Dropout probability in transformer decoder. Default: 0.0
        activation (Callable[..., nn.Module]): Activation function of transformer
            decoder. Default: nn.GELU
        layer_norm_eps (float): Epsilon value for transformer decoder layer norms.
            Default: 1e-5
        norm_first (bool): Whether to apply layer normalization before or after
            self-attention in transformer decoder. Default: True
        final_layer_norm_eps (Optional[float]): Regularization value for final layer norm
            in transformer decoder. Default: 1e-5
        visual_embedding_dim (Optional[int]): Dimension of visual embeddings inside
            transformer decoder (used for cross-attention). Default: None (visual
            embeddings assumed to be same dimension as text embeddings)
    """

    def __init__(
        self,
        input_seq_len: int,
        text_embedding_dim: int,
        n_layer: int,
        n_head: int,
        dim_feedforward: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        final_layer_norm_eps: Optional[float] = 1e-5,
        visual_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.transformer_decoder = TransformerDecoder(
            n_layer=n_layer,
            d_model=text_embedding_dim,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            use_cross_attention=True,
            final_layer_norm_eps=final_layer_norm_eps,
            dim_kv=visual_embedding_dim,
        )
        if output_dim is not None:
            self.output_projection = nn.Linear(
                text_embedding_dim, output_dim, bias=False
            )
        else:
            self.output_projection = None

        self.register_buffer(
            "causal_mask",
            get_causal_attention_mask(input_seq_len).to(dtype=torch.bool),
            persistent=False,
        )

    def forward(self, texts: Tensor, images: Tensor) -> Tensor:
        """
        Args:
            texts (Tensor): Tensor containing text embeddings of shape [batch_size, text_seq_length, embeddings_dim]
            images (Tensor): Tensor containing image embeddings of shape [batch_size, image_seq_length, embeddings_dim]
            text_causal_mask (Tensor): Tensor containing causal mask of shape [text_seq_length, text_seq_length]
        Returns:
        Tensor: Tensor containing output embeddings of shape [batch_size, text_seq_length, output_dim]
        """
        seq_len = texts.shape[1]
        assert self.causal_mask.shape == (seq_len, seq_len)
        decoder_outputs = self.transformer_decoder(
            hidden_states=texts,
            encoder_hidden_states=images,
            attention_mask=self.causal_mask,
        )
        hidden_states = decoder_outputs.last_hidden_state
        assert hidden_states is not None, "hidden states must not be None"
        if self.output_projection is not None:
            out = self.output_projection(hidden_states)
        else:
            out = hidden_states
        return out
