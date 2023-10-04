# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Tuple, Union

import torch

from torch import nn, Tensor
from torchmultimodal.modules.layers.patch_embedding import PatchEmbeddings
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerOutput,
)
from torchmultimodal.utils.common import load_module_from_url


class VisionTransformer(nn.Module):
    """
    General image transformer encoder with embeddings. Similar to ``VisionTransformer`` in torchvision,
    but more composable. Can be constructed with any user-provided embeddings, encoder, and task head.

    Attributes:
        embeddings (nn.Module): Module that projects image pixels into embeddings.
            See :py:class: PatchEmbeddings for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class: TransformerEncoder for interface.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: init_transformer_weights as an example. Defaults to ``None``.

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
        pooler: Optional[nn.Module] = None,
        weight_init_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self.embeddings = embeddings
        self.encoder = encoder
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
        ).embeddings

        encoder_output = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            return_hidden_states=True,
        )
        last_hidden_state = encoder_output.last_hidden_state

        if self.pooler is not None:
            assert (
                last_hidden_state is not None
            ), "For pooler, last hidden state cannot be None."
            pooled_output = self.pooler(last_hidden_state)
        else:
            pooled_output = None

        return TransformerOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class GlobalAveragePooler(nn.Module):
    """
    Global average pooler that averages the embeddings over all the patches in a sample
    and applies layer norm and an optional linear head on top.
    Args:
        input_dim (int): hidden dim of the transformer last hidden state.
        output_dim (Optional[int]): output dim of the linear head. if None, no linear head is added. Defaults to None.
        ln_eps (float): layer norm epsilon. Defaults to 1e-6.
        init_weights (Optional[Callable]): function to initialize weights of the module. Defaults to None.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        ln_eps: float = 1e-6,
        init_weights: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=ln_eps)
        if output_dim:
            self.head: nn.Module = nn.Linear(input_dim, output_dim)
        else:
            self.head = nn.Identity()
        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with shape bsz x len x input_dim. The first entry in assumed to be CLS
                and ignored during averaging
        Returns:
            Tensor: Output tensor with shape bsz x output_dim
        """
        out = x[:, 1:, :].mean(dim=1)
        out = self.norm(out)
        out = self.head(out)
        return out


def vision_transformer(
    *,
    patch_size: int,
    hidden_dim: int,
    dim_feedforward: int,
    n_layer: int,
    n_head: int,
    image_size: Union[int, Tuple[int, int]] = 224,
    num_channels: int = 3,
    activation: Callable[..., nn.Module] = nn.GELU,
    transformer_dropout: float = 0.0,
    patch_embed_dropout_prob: float = 0.0,
    layer_norm_eps: float = 1e-6,
    final_layer_norm_eps: Optional[float] = 1e-6,
    norm_first: bool = True,
    include_cls_embed: bool = True,
    drop_path_rate: Optional[float] = None,
    patch_drop_rate: Optional[Union[float, Tuple[float, float]]] = None,
    pooler: Optional[nn.Module] = None,
    ckpt_path: str = None,
) -> VisionTransformer:
    """
    Args:
        patch_size (int): Size of each patch that the image is divided into.
        hidden_dim (int): Hidden dimension of the output of patch embedding and input to the transformer
        dim_feedforward (int): Dimension of the feedforward network inside the transformer.
        n_layer (int): Number of hidden layers in the transformer.
        n_head (int): Number of attention heads in the transformer.
        image_size (Union[int, Tuple[int, int]]): Size of the input image. If tuple, should be height, width. \
            If int, square input is assumed. Defaults to 224
        num_channels (int): Number of channels in the input. Defaults to 3.
        activation (Callable): Activation function for the transformer. Defaults to nn.GELU.
        transformer_dropout (float): Dropout probability for the transformer. Defaults to 0.0.
        patch_embed_dropout_prob (float): Dropout probability for the patch embedding. Defaults to 0.0.
        layer_norm_eps (float): layer norm epsilon for the transformer blocks. Defaults to 1e-6.
        final_layer_norm_eps (Optional[float]) = layer norm epsilon for final ln layer of transformer. Defaults to 1e-6.
        norm_first(bool): indicates whether layer norm is applied before or after self attention in the transformer block.
        Defaults to True for vits
        include_cls_embed (bool): whether to add cls token inside of image embeddings. Defaults to True
        drop_path_rate (Optional[float]): use stochastic drop path instead of dropout for attn and feedforward dropout
        in transformer block. Defaults to None.
        patch_drop_rate (Optional[Union[float, Tuple[float, float]]]): ratio of patches to mask out before passing through encoder
        as in https://arxiv.org/abs/2212.00794. Set to tuple if dimension wise masking is needed (for 2d masking). Defaults to None.
        pooler (nn.Module, optional): Pooling function to be applied to the last hidden state from the transformer like avg pooling.
        Defaults to None
    """
    image_embedding = PatchEmbeddings(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_dim,
        hidden_dropout_prob=patch_embed_dropout_prob,
        patch_drop_rate=patch_drop_rate,
        num_channels=num_channels,
        include_cls_embed=include_cls_embed,
    )
    transformer_encoder = TransformerEncoder(
        n_layer=n_layer,
        d_model=hidden_dim,
        n_head=n_head,
        dim_feedforward=dim_feedforward,
        dropout=transformer_dropout,
        activation=activation,
        layer_norm_eps=layer_norm_eps,
        norm_first=norm_first,
        final_layer_norm_eps=final_layer_norm_eps,
        drop_path_rate=drop_path_rate,
    )
    vit = VisionTransformer(
        embeddings=image_embedding, encoder=transformer_encoder, pooler=pooler
    )
    if ckpt_path:
        load_module_from_url(vit, ckpt_path)
    return vit


def vit_b_16(pooler: Optional[nn.Module] = None, **kwargs: Any) -> VisionTransformer:
    return vision_transformer(
        patch_size=16,
        n_layer=12,
        n_head=12,
        hidden_dim=768,
        dim_feedforward=3072,
        pooler=pooler,
        **kwargs,
    )


def vit_b_32(pooler: Optional[nn.Module] = None, **kwargs: Any) -> VisionTransformer:
    return vision_transformer(
        patch_size=32,
        n_layer=12,
        n_head=12,
        hidden_dim=768,
        dim_feedforward=3072,
        pooler=pooler,
        **kwargs,
    )


def vit_l_16(pooler: Optional[nn.Module] = None, **kwargs: Any) -> VisionTransformer:
    return vision_transformer(
        patch_size=16,
        n_layer=24,
        n_head=16,
        hidden_dim=1024,
        dim_feedforward=4096,
        pooler=pooler,
        **kwargs,
    )


def vit_l_32(pooler: Optional[nn.Module] = None, **kwargs: Any) -> VisionTransformer:
    return vision_transformer(
        patch_size=32,
        n_layer=24,
        n_head=16,
        hidden_dim=1024,
        dim_feedforward=4096,
        pooler=pooler,
        **kwargs,
    )


def vit_h_14(pooler: Optional[nn.Module] = None, **kwargs: Any) -> VisionTransformer:
    return vision_transformer(
        patch_size=14,
        n_layer=32,
        n_head=16,
        hidden_dim=1280,
        dim_feedforward=5120,
        pooler=pooler,
        **kwargs,
    )
