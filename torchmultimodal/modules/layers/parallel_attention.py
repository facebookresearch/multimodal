# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from torchmultimodal.modules.layers.normalizations import RMSNorm

# from position_embedding import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """expands kv_heads to match q num_heads
    via
    torch.repeat_interleave(x, dim=2, repeats=n_rep)"""

    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class ParallelAttentionBlock(nn.Module):
    """
    Transformer layer multi-head attention and MLP, in a parallelized fashion rather than sequential,
    with optional attention masking.
    Inspired by PaLM:  https://arxiv.org/abs/2204.02311

    * We use SwiGLU for the activation function
    * SwiGLU will approximate same total num params as traditional MLP with GELU
    * Cross Attention is not enabled here

    * MQA and GQA are enabled - modify heads via 'num_heads_group_query_attn'
    * MQA is num_heads_group_query_attn = 1
    * GQA is num_heads_group_query_attn < num_heads, and must be evenly divisible into num_heads

    * Bias is enabled by default.  Experiment with removing via use...bias = False, for your application.

    * Parallel blocks have automated weight initialization via _init_weights.
    * Please pass in num_layers of your model in num_layers for the weight initialization.
    """

    def __init__(
        self,
        emb_dimension: int,
        num_heads: int,
        head_dimension: int = None,
        mlp_expansion_ratio: float = 2.6875,  # 8/3 is param matching
        qk_normalization: bool = True,
        projection_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_group_query_attention: bool = True,
        num_heads_group_query_attention: int = 1,
        use_in_projection_bias: bool = True,
        use_out_projection_bias: bool = True,
        use_weight_init: bool = True,
        num_layers: int = 0,
        use_rms_norm: bool = True,
        use_rotary_embeddings: bool = False,
        max_expected_seq_len: int = 2048,  # needed only if using rotary
    ) -> None:
        super().__init__()

        version_check = not version.parse(torch.__version__) < version.parse("2.0.0")
        assert (
            version_check
        ), f"Parallel Attention Blocks requires PT 2.0+, you are running {torch.__version__}.\nPlease upgrade your PyTorch version."

        self.num_heads = num_heads
        self.emb_dim = emb_dimension
        self.head_dim = head_dimension if head_dimension else emb_dimension // num_heads
        assert (
            self.emb_dim % self.num_heads == 0
        ), f"dimensions {self.emb_dim.shape} must be evenly divisible by num_heads {num_heads=}"

        # group query attn
        if use_group_query_attention:
            assert (
                self.num_heads % num_heads_group_query_attention == 0
            ), f"{self.num_heads=} not evenly divisible by {num_heads_group_query_attention=}"

        self.use_variable_kv = use_group_query_attention
        self.group_num_kv = num_heads_group_query_attention
        self.num_kv = self.group_num_kv if self.use_variable_kv else self.num_heads
        self.kv_head_dims = self.head_dim * self.num_kv
        self.kv_expansion_factor = int(self.num_heads / self.group_num_kv)
        assert (
            self.kv_expansion_factor > 0
        ), f"kv expansion factor must be positive integer, got {self.kv_expansion_factor=}"

        self.mlp_hidden_dim = int(mlp_expansion_ratio * self.emb_dim)

        self.qk_norm: bool = qk_normalization

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.mlp_dropout = nn.Dropout(projection_dropout)

        # weight init
        self.num_layers = num_layers
        self.use_weight_init = use_weight_init
        if self.use_weight_init:
            assert (
                self.num_layers > 0
            ), f"Need to pass in global num layers for weight init, {self.num_layers=}"

        self.weight_init_standard_dev = 0.02 / math.sqrt(2 * self.num_layers)

        self.use_in_projection_bias = use_in_projection_bias
        self.use_out_projection_bias = use_out_projection_bias

        # previous init params, moved to internal defaults for streamlining
        normalization_layer = RMSNorm if use_rms_norm else nn.LayerNorm
        self.mlp_activation = nn.SiLU()

        self.num_q = 1

        self.in_proj_dims = [
            self.head_dim * num_heads * self.num_q,
            self.kv_head_dims,
            self.kv_head_dims,
            self.mlp_hidden_dim,
            self.mlp_hidden_dim,
        ]  # q, k, v, mlp, gate

        # layer objects
        self.in_norm = normalization_layer(emb_dimension)
        self.in_proj = nn.Linear(
            emb_dimension, sum(self.in_proj_dims), bias=use_in_projection_bias
        )

        self.q_norm = normalization_layer(self.head_dim)
        self.k_norm = normalization_layer(self.head_dim)

        # fused out projection
        fused_out_input_dim = emb_dimension + self.mlp_hidden_dim
        self.out_fused_proj = nn.Linear(
            fused_out_input_dim, emb_dimension, bias=use_out_projection_bias
        )

        # rotary embeddings
        if use_rotary_embeddings:
            raise AssertionError("RotaryEmbeddings has not been merged yet...")
            # self.rotary_emb = RotaryEmbedding(emb_dimension, max_expected_seq_len)
        else:
            self.rotary_emb = None

        # init weights
        if use_weight_init:
            self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """init weights using trunc + llama style depth scaling"""
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                # std_dev = 0.02 / math.sqrt(2 * self.num_layers)
                std=self.weight_init_standard_dev,
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        cross_x: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        rel_pos_bias: Optional[torch.Tensor] = None,
        has_causal_mask: bool = False,
    ) -> torch.Tensor:
        """TODO:   No KV cache support yet"""

        assert not (
            rel_pos_bias is not None and self.rotary_emb is not None
        ), "Rotary and additive biases are exclusive"
        assert not (
            (rel_pos_bias is not None or mask is not None) and has_causal_mask
        ), "Causal mask optimization only valid without attn_mask or rel_pos_bias"

        batch_size, seq_len, channels = x.shape

        y = self.in_norm(x)
        y = self.in_proj(y)

        q, k, v, inner_mlp, gate = torch.split(y, self.in_proj_dims, dim=-1)

        # b n nq h d
        q = q.view(batch_size, seq_len, self.num_q, self.num_heads, self.head_dim)

        q = q[:, :, 0].transpose(2, 1)

        if self.rotary_emb:
            start_pos = 0  # TODO: No kv-cache yet, when that happens this is seqlen saved in kv-cache
            q, k = self.rotary_emb(q, k, start_pos)

        # group query expansion
        def kv_expansion(head: torch.Tensor) -> torch.Tensor:
            head = head.view(
                batch_size, seq_len, self.num_kv, self.head_dim
            )  # b n hnum dimh
            # bs, slen, n_kv_heads, head_dim = x.shape
            if self.use_variable_kv and self.num_kv > 1:
                head = repeat_kv(head, n_rep=self.kv_expansion_factor)
            return head.transpose(2, 1)  # b hnum n dimh

        k = kv_expansion(k)
        v = kv_expansion(v)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Merge rel pos bias and mask into single float mask
        if rel_pos_bias is None:
            # Given SDPA API, we expect users to either provide a boolean mask if
            # they expect masked_fill to be done inside SDPA, or provide the float
            # mask already with the correct -inf
            attn_mask = mask  # b? ...? nq nk
        else:
            attn_mask = rel_pos_bias  # b? ...? nq nk

            # We expect the shapes of mask and rel_pos_bias to be at least broadcastable
            if mask is not None:
                # Can't do in-place op in case broadcast makes attn_mask bigger
                attn_mask = attn_mask.masked_fill(mask == 0, -float("inf"))

        final_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attention_dropout.p,
            is_causal=has_causal_mask,
        )

        final_attn = (
            final_attn.transpose(2, 1)
            .contiguous()
            .view(batch_size, seq_len, self.head_dim * self.num_heads)
        )

        # swiglu
        activated_mlp = self.mlp_activation(inner_mlp) * gate

        if self.mlp_dropout.p:
            activated_mlp = self.mlp_dropout(activated_mlp)

        y = torch.cat((final_attn, activated_mlp), dim=2)

        y = self.out_fused_proj(y)

        # Add residual
        x = x + y
        return x
