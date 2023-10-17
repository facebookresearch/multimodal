# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.activation import GEGLU
from torchmultimodal.modules.layers.multi_head_attention import (
    MultiHeadAttentionWithCache,
)
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm, Fp32LayerNorm
from torchmultimodal.utils.common import init_module_parameters_to_zero


class SpatialTransformerCrossAttentionLayer(nn.Module):
    """Transformer encoder layer with cross-attention mechanism. This layer contains
    2 attention blocks that use PyTorch's scaled dot product attention. The first attention
    block performs self attention block, while the second performs cross-attention. If
    `context_dim` is not set or `context` is not passed, the second block defaults to
    self attention.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L196

    Attributes:
        d_model (int): size of hidden dimension of input
        num_heads (int): number of attention heads
        context_dim (int, optional): size of context embedding. If None,
            use self attention. Defaults to None.
        dropout (float): Dropout to apply post attention layers.
            Defaults to 0.
        attention_dropout (float): Dropout to apply to scaled dot product
            attention. Defaults to 0.

    Args:
        x (Tensor): input Tensor of shape [b, seq_len, d_model]
        context (Tensor, optional): Context tensor of shape
            [b, seq_len, context_dim]. Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        # Optional context is added only for parity.
        # TODO: Remove optional context if this code path is
        # not used by genai use cases
        self.use_context = context_dim is not None
        self.self_attn_layernorm = Fp32LayerNorm(d_model)
        self.self_attn = MultiHeadAttentionWithCache(
            dim_q=d_model,
            dim_kv=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            add_bias=False,
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        # If no context is passed, then cross-attention blocks end up performing self-attention
        self.cross_attn_layernorm = Fp32LayerNorm(d_model)
        self.cross_attn = MultiHeadAttentionWithCache(
            dim_q=d_model,
            # defaults to self attention if context dim not provided
            dim_kv=context_dim if context_dim else d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            add_bias=False,
        )
        self.cross_attn_dropout = nn.Dropout(dropout)

        # scaling the projection dimension by 4 to match the logic in the original implementation
        projection_dim = d_model * 4
        self.feed_forward_block = nn.Sequential(
            Fp32LayerNorm(d_model),
            # Projection dim is scaled by 2 to perform the GELU operation which chunks the
            # input projection into 2 parts and combines them to obtain the activation.
            nn.Linear(d_model, projection_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, d_model),
        )

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        if not self.use_context:
            context = None

        h = self.self_attn_layernorm(x)
        h = self.self_attn_dropout(self.self_attn(query=h, key=h, value=h)) + x
        h_res = h
        h = self.cross_attn_layernorm(h)
        context = context if context is not None else h
        h = (
            self.cross_attn_dropout(
                self.cross_attn(query=h, key=context, value=context)
            )
            + h_res
        )
        h = self.feed_forward_block(h) + h
        return h


class SpatialTransformer(nn.Module):
    """Transformer block with cross-attention mechanism that operates on
    image-like data. First, it flattens the spatial dimensions of the image
    to shape (batch_size x (height * width) x num_channels) and applies an
    input projection. Next, the projected input is passed ta block of stacked
    transformer cross attention layers. Finally, the output goes through another
    projection and is reshaped back to an image.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L218

    Attributes:
        in_channls(int): number of channels in input image.
        num_heads (int): number of attention heads.
        num_layers (int): number of transformer encoder layers.
        context_dims (Sequence[int], optional): Size of context embedding for
            every transformer layer. If len(context_dim) < num_layers, expand context_dims
            to have length num_layers. If None, use self attention for every layer.
            Defaults to None.
        use_linear_projections (bool, optional): If True, use linear input and output
         projections instead of 1x1 conv projections. Defaults to False.
        dropout (float): Dropout to apply post attention layers.
            Defaults to 0.
        attention_dropout (float): Dropout to apply to scaled dot product
            attention. Defaults to 0.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.

    Args:
        x (Tensor): input Tensor of shape [b, seq_len, d_model]
        context (Sequence[Tensor], optional): List of context tensors of shape
            [b, seq_len, context_dim] each. Must be equal to the number of
            transformer layers. Defaults to None.

    Raises:
        ValueError: If `num_layers` is not a multiple of length of `context_dims`.
        RuntimeError: If `len(self.transformer_layers)` is not a multiple of length of `context`.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        num_layers: int,
        context_dims: Optional[Sequence[int]] = None,
        use_linear_projections: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_groups: int = 32,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        # Optional context is added only for parity with oss implementation.
        # TODO: Remove optional context if this code path is
        # not used by genai use cases
        self.use_context = context_dims is not None
        if context_dims is None:
            context_dims = [context_dims] * num_layers
        elif num_layers != len(context_dims):
            assert isinstance(context_dims, list)
            if num_layers % len(context_dims) != 0:
                raise ValueError(
                    "`num_layers` must be a multiple of the length of `context_dims`."
                )

            repeating_factor = int(num_layers / len(context_dims))
            print(
                f"WARNING: context dims {context_dims} of length {len(context_dims)} does not match "
                f"'num_layers'={num_layers}. Expanding context_dims to {context_dims * repeating_factor}."
            )
            context_dims = context_dims * repeating_factor

        self.use_linear_projections = use_linear_projections

        self.norm = Fp32GroupNorm(
            num_groups=norm_groups, num_channels=in_channels, eps=norm_eps
        )

        # Initialize input and output projections. If using linear projections, both
        # projections are initialized with nn.Linear, otherwise use 1x1 convolutions.
        if self.use_linear_projections:
            self.in_projection = nn.Linear(in_channels, in_channels)
            self.out_projection = nn.Linear(in_channels, in_channels)
        else:
            self.in_projection = nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0
            )
            self.out_projection = nn.Conv2d(
                in_channels, in_channels, kernel_size=1, stride=1, padding=0
            )
        # Initialize out projection with zero weight and bias. This helps with
        # training stability. Initialization trick from Fixup Initialization.
        # https://arxiv.org/abs/1901.09321
        init_module_parameters_to_zero(self.out_projection)

        self.transformer_layers = nn.ModuleList(
            [
                SpatialTransformerCrossAttentionLayer(
                    d_model=in_channels,
                    num_heads=num_heads,
                    context_dim=context_dims[i],
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        context: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        # If context_dims were not provided in init, default to self attention
        # by setting passed context to None.
        # TODO: Remove this logic if there is no case where context or context dims are None.
        if not self.use_context:
            context = None
        if isinstance(context, Sequence) and len(context) != len(
            self.transformer_layers
        ):
            if len(self.transformer_layers) % len(context) != 0:
                raise RuntimeError(
                    "`len(self.transformer_layers)` must be a multiple of the length of `context`."
                )

            print(
                f"WARNING: context of length {len(context)} does not match 'num_layers'={len(self.transformer_layers)}."
                f" Contexts will be re-used {int(len(self.transformer_layers)/len(context))} times."
            )

        _, _, H, W = x.shape
        h = self.norm(x)
        # For linear projection, first reshape and then apply projection
        if self.use_linear_projections:
            # b * c * h * w -> b * (h * w) * c
            h = torch.transpose(torch.flatten(h, start_dim=2), 1, 2)
            h = self.in_projection(h)
        else:  # For conv projection, first apply projection and then reshape
            h = self.in_projection(h)
            # b * c * h * w -> b * (h * w) * c
            h = torch.transpose(torch.flatten(h, start_dim=2), 1, 2)

        for i in range(len(self.transformer_layers)):
            if isinstance(context, Sequence):
                _context = context[i % len(context)]
            else:
                _context = None

            h = self.transformer_layers[i](h, context=_context)

        # For linear projection, first apply projection and then unflatten
        if self.use_linear_projections:
            h = self.out_projection(h)
            h = torch.unflatten(torch.transpose(h, 1, 2), dim=2, sizes=(H, W))
        else:  # For conv projection, first unflatten and then apply projection
            h = torch.unflatten(torch.transpose(h, 1, 2), dim=2, sizes=(H, W))
            h = self.out_projection(h)

        return h + x
