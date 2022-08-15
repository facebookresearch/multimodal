# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import TransformerOutput


class VisionTransformer(nn.Module):
    """
    General image transformer encoder with embeddings. Similar to ``VisionTransformer`` in torchvision,
    but more composable. Can be constructed with any user-provided embeddings, encoder, and task head.

    Attributes:
        embeddings (nn.Module): Module that projects image pixels into embeddings.
            ``forward()`` should follow interface:
                images: Optional[Tensor], input data
                image_patches_mask: Optional[Tensor], mask for patch embeddings
        encoder (nn.Module): Module for transformer encoder. ``forward()`` should follow interface:
            Inputs:
                hidden_states: Tensor, input for encoder
                attention_mask: Optional[Tensor], shape ``(b, num_heads, query_seq_len, key_seq_len)``
                return_attn_weights: bool. See ``TransformerEncoder``.
                return_hidden_states: bool. See ``TransformerEncoder``.
            Returns:
                ``TransformerOutput``
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder, if provided.
        pooler (nn.Module, optional): Module for head to be applied after layernorm, if provided.

    Args:
        images (Tensor): Tensor of input images of shape ``(b, c, h, w)``.
        image_patches_mask (Tensor, optional): Tensor indicating which patches to replace with mask tokens,
            shape ``(b, seq_len)``, where seq_len = (image_size // patch_size) ** 2
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape ``(b, seq_len + 1)``.
            Concatenating class_token adds 1 to seq_len.
    """

    def __init__(
        self,
        embeddings: nn.Module,
        encoder: nn.Module,
        layernorm: Optional[nn.Module] = None,
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler

        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(
        self,
        images: Tensor,
        image_patches_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> TransformerOutput:

        embedding_output = self.embeddings(
            images, image_patches_mask=image_patches_mask
        )

        encoder_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            return_attn_weights=True,
            return_hidden_states=True,
        )
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return TransformerOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )
