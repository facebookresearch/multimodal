# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable

from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import TransformerCrossAttentionLayer
from torchmultimodal.utils.attention import get_extended_attention_mask


class ALBEFMultimodalEncoder(nn.Module):
    """
    Construct multimodal embeddings from image embeddings, text embeddings, and text attention mask.

    The ALBEFMultimodalEncoder is similar to ALBEFTextEncoder, with the addition of image-text cross attention in encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers. Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder. Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder. Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        layer_norm_eps (float): The epsilon used by the layer normalization layers. Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer. Default is GELU.

    Inputs:
        image_embeds (Tensor of size (batch_size, image_seq_length, hidden_size)): Image embeddings from a vision encoder.
        text_embeds (Tensor of size (batch_size, text_seq_length, hidden_size)): Text embeddings from a text encoder.
        text_atts (Tensor of size (batch_size, text_seq_length)): Mask to avoid performing attention on padding token indices.
            Mask values selected in [0, 1]: 1 for tokens that are NOT MASKED, 0 for MASKED tokens.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [
                TransformerCrossAttentionLayer(
                    d_model=hidden_size,
                    n_head=num_attention_heads,
                    dim_feedforward=intermediate_size,
                    activation=transform_act_fn,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        image_embeds: Tensor,
        text_embeds: Tensor,
        text_atts: Tensor,
    ) -> Tensor:
        text_atts = get_extended_attention_mask(text_atts)
        hidden_states = text_embeds
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                image_embeds,
                attention_mask=text_atts,
            )
        return hidden_states
