# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.attention import MultiHeadAttention, split_multihead
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm
from torchmultimodal.utils.common import shift_dim


class ADMAttentionBlock(nn.Module):
    """Attention block in the ADM net that consists of group norm, multihead attention, and a residual connection.

    Follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233)

    Code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L259

    Defaults taken from https://fburl.com/code/tiryi2f9.

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of q, k, v in attention module.
            Needs to be divisible by norm_groups.
        dim_cond (Optional[int]): dimensionality of conditional input for cross attention. If not specified,
            do not use conditional input.
        rescale_skip_connection (bool): whether to rescale skip connection by 1/sqrt(2), as described in "Diffusion
            Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233). Defaults to False.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        conditional_embedding (Optional[Tensor]): tokens of shape [b, n, dim_cond] where n is the number of tokens.
            If provided, will be passed as cross-attention input to the MultiHeadAttention block. Defaults to None.
    """

    def __init__(
        self,
        num_channels: int,
        dim_cond: Optional[int] = None,
        num_heads: int = 1,
        rescale_skip_connection: bool = False,
        norm_groups: int = 32,
    ):
        super().__init__()

        if num_channels % norm_groups != 0:
            raise ValueError("Channel dims need to be divisible by norm_groups")

        self.norm = Fp32GroupNorm(norm_groups, num_channels)
        self.attn = adm_attention(num_channels, dim_cond, num_heads)
        self.rescale_skip_connection = rescale_skip_connection

    def forward(
        self,
        x: Tensor,
        conditional_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        norm_out = self.norm(x)
        # [b, c, h, w] -> [b, h, w, c]
        norm_out = shift_dim(norm_out, 1, -1)
        attn_out = self.attn(norm_out, conditional_embedding=conditional_embedding)
        # [b, h, w, c] -> [b, c, h, w]
        attn_out = shift_dim(attn_out, -1, 1)
        if self.rescale_skip_connection:
            return (x + attn_out) / 1.414
        else:
            return x + attn_out


class ADMCrossAttention(nn.Module):
    """Similar to standard cross-attention, except conditioning inputs are passed through a separate projection
    and then concatenated with the key and value vectors before scaled dot product attention.

    Code ref: https://fburl.com/code/rxl1md57

    Attributes:
        dim_qkv (int): embedding dimension of query, key, and value vectors. conditional_embedding is projected into this
            dimension * 2, to account for k and v.
        dim_cond (int, optional): embedding dimension of conditional input. If unspecified, this class becomes standard
            self attention.

    Args:
        q, k, v (Tensor): Query/key/value of shape [b, h, d1, ..., dn, dim_qkv // h] or [b, h, seq_len, dim_qkv //h] where
            h is number of attention heads, d1, ..., dn are spatial dimensions and dim_qkv is
            the embedding dim.
        conditional_embedding (Tensor, Optional): tensor of shape [b, d1, ..., dn, dim_cond] to condition k and v on

    Returns:
        A tensor of shape [b, h, d1, ..., dn, dim_qkv // h] with the output of the attention calculation.
    """

    def __init__(self, dim_qkv: int, dim_cond: Optional[int] = None) -> None:
        super().__init__()
        self.dim_qkv = dim_qkv
        self.cond_proj: Optional[nn.Module] = None
        if dim_cond is not None:
            # Times 2 for both k and v
            self.cond_proj = nn.Linear(dim_cond, dim_qkv * 2)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        conditional_embedding: Optional[Tensor] = None,
    ) -> Tensor:
        _, n_head, *spatial_dims, dim_q = q.shape
        dim_q = dim_q * n_head
        dim_k = k.shape[-1] * n_head
        dim_v = v.shape[-1] * n_head
        if self.dim_qkv != dim_q or self.dim_qkv != dim_k or self.dim_qkv != dim_v:
            raise ValueError(
                f"The embedding dim of q, k, v does not match expected embedding dim of {self.dim_qkv}."
            )
        if self.dim_qkv % n_head != 0:
            raise ValueError(
                "The embedding dim of q, k, v must be a multiple of the number of attention heads."
            )

        # [b, h, d1, ..., dn, dim_qkv] -> [b, h, t, dim_qkv]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if conditional_embedding is not None and self.cond_proj is not None:
            # [b, d1, ..., dn, dim_cond] -> [b, h, d1, ..., dn, dim_qkv * 2 // h]
            cond = split_multihead(self.cond_proj(conditional_embedding), n_head)
            # [b, h, d1, ..., dn, dim_qkv * 2 // h] -> [b, h, t, dim_qkv * 2 // h]
            cond = cond.flatten(start_dim=2, end_dim=-2)
            # [b, h, t, dim_qkv // h]
            cond_k, cond_v = cond.split(self.dim_qkv // n_head, dim=-1)
            # concat on sequence length dimension
            k = torch.cat([cond_k, k], dim=2)
            v = torch.cat([cond_v, v], dim=2)

        # Use PyTorch's optimized scaled_dot_product_attention
        attn = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0,
            is_causal=False,
        )
        # [b, h, t, dim_qkv // h] -> [b, h, d1, ..., dn, dim_qkv // h]
        attn = attn.unflatten(2, spatial_dims)

        return attn


def adm_attention(
    num_channels: int,
    dim_cond: Optional[int] = None,
    num_heads: int = 1,
) -> nn.Module:
    attn = ADMCrossAttention(
        dim_qkv=num_channels,
        dim_cond=dim_cond,
    )
    return MultiHeadAttention(
        dim_q=num_channels,
        dim_kv=num_channels,
        n_head=num_heads,
        attn_module=attn,
    )


def adm_attn_block(
    num_channels: int,
    dim_cond: Optional[int] = None,
) -> ADMAttentionBlock:
    return ADMAttentionBlock(
        num_channels=num_channels,
        dim_cond=dim_cond,
    )
