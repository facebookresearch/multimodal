# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.layers.transformer import TransformerDecoder
from torchmultimodal.utils.attention import get_causal_attention_mask


class CoCaTextEmbeddings(nn.Module):
    """
    Text embeddings for CoCa model. Includes token embeddings, positional embeddings,
    and optional CLS embedding.

    Args:
        vocab_size (int): Size of the vocab
        num_positions (int): Number of token positions for positional embeddings
            not including cls.
        embedding_dim (int): Output embedding dimension
        pad_idx (Optional[int]): Padding index to be ignored by token embeddings.
            Default: 0
        embed_cls (bool): Whether to include CLS embedding. Default: True
    """

    def __init__(
        self,
        vocab_size: int,
        num_positions: int,
        embedding_dim: int,
        pad_idx: Optional[int] = 0,
        embed_cls: bool = True,
    ):
        super().__init__()
        self.num_positions = num_positions
        if embed_cls:
            self.cls_embedding = nn.Parameter(torch.empty(embedding_dim))
        else:
            self.cls_embedding = None

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.position_embeddings = nn.Parameter(
            torch.empty(num_positions, embedding_dim)
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.01)
        if self.cls_embedding is not None:
            nn.init.constant_(self.cls_embedding, 0.01)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids (Tensor of size (batch_size, seq_length)):
                Indices of input sequence tokens.
        Returns:
            Tensor of size (batch_size, seq_length, embedding_dim)
        """

        assert input_ids.shape[1] == (
            self.num_positions if self.cls_embedding is None else self.num_positions - 1
        )
        embeddings = self.token_embeddings(input_ids)

        if self.cls_embedding is not None:
            # Expand cls embedding (embedding_dim) -> (batch_size, 1, embedding_dim)
            cls_embed = self.cls_embedding.reshape(1, 1, -1).repeat(
                input_ids.shape[0], 1, 1
            )
            embeddings = torch.cat([embeddings, cls_embed], dim=1)

        embeddings = embeddings + self.position_embeddings.to(dtype=embeddings.dtype)

        return embeddings


class CoCaTextDecoder(nn.Module):
    """
    Text decoder for CoCa model.
    Based on the implementation in open_clip: https://tinyurl.com/2jswrb9h

    Args:
        vocab_size (int): Size of the vocab
        num_positions (int): Number of token positions for positional embeddings.
        embedding_dim (int): Embedding dimension for transformer
        n_layer (int): Number of transformer layers
        n_head (int): Number of attention heads
        dim_feedforward (int): Hidden dimension in transformer FFN
        output_dim (int): Output dimension of decoder cls / eos projection
        pad_idx (Optional[int]): Padding index (will be masked from CLS token).
            Default: 0
        embed_cls (bool): Whether to append CLS embedding. Default: True
        dropout (float): Dropout probability in transformer decoder
        activation (Callable[..., nn.Module]): Activation function of transformer
            decoder. Default: nn.GELU
        layer_norm_eps (float): Epsilon value for transformer decoder layer norms.
            Default: 1e-5
        norm_first (bool): Whether to apply layer normalization before or after
            self-attention in transformer decoder. Default: True
        final_layer_norm_eps (Optional[float]): Final layer norm epsilon. Only applied
            to CLS token if embed_cls=True. Default: 1e-5
    """

    def __init__(
        self,
        vocab_size: int,
        num_positions: int,
        embedding_dim: int,
        n_layer: int,
        n_head: int,
        dim_feedforward: int,
        output_dim: int,
        pad_idx: Optional[int] = 0,
        embed_cls: bool = True,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        final_layer_norm_eps: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_cls = embed_cls
        self.num_positions = num_positions
        self.embeddings = CoCaTextEmbeddings(
            vocab_size=vocab_size,
            num_positions=num_positions,
            embedding_dim=embedding_dim,
            pad_idx=pad_idx,
            embed_cls=embed_cls,
        )
        self.transformer_decoder = TransformerDecoder(
            n_layer=n_layer,
            d_model=embedding_dim,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            use_cross_attention=False,
        )
        if final_layer_norm_eps is not None:
            self.ln_final = nn.LayerNorm(
                normalized_shape=embedding_dim, eps=final_layer_norm_eps
            )
        self.text_projection = nn.Linear(embedding_dim, output_dim, bias=False)
        self.register_buffer(
            "causal_mask",
            get_causal_attention_mask(num_positions).to(dtype=torch.bool),
            persistent=False,
        )
        self.init_parameters(embedding_dim, n_layer)

    def init_parameters(self, embedding_dim: int, n_layer: int) -> None:
        # Initialization based on https://tinyurl.com/cmm7cwjt
        attn_std = embedding_dim**-0.5
        proj_std = (2 * embedding_dim * n_layer) ** -0.5
        fc_std = (2 * embedding_dim) ** -0.5
        for layer in self.transformer_decoder.layer:
            nn.init.normal_(layer.attention.q_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.k_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.v_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.output_proj.weight, std=proj_std)
            nn.init.normal_(layer.feedforward.model[0].weight, std=fc_std)
            nn.init.normal_(layer.feedforward.model[2].weight, std=proj_std)
        nn.init.normal_(self.text_projection.weight, std=embedding_dim**0.5)

    def build_mask(
        self,
        input_ids: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # If no CLS token, we can directly return the causal mask
        if not self.embed_cls or self.pad_idx is None:
            return self.causal_mask

        # If padding_mask is not passed, infer it
        if padding_mask is None:
            padding_mask = input_ids != self.pad_idx
        assert padding_mask is not None
        # (batch_size, seq_len) -> (batch_size, 1, seq_len)
        padding_mask = padding_mask.unsqueeze(1)

        # (batch_size, 1, seq_len) -> (batch_size, seq_len+1, seq_len+1)
        padding_mask = F.pad(padding_mask, (1, 0, padding_mask.shape[2], 0), value=1.0)
        # Make broadcastable for MHA
        mask = (padding_mask * self.causal_mask).unsqueeze(1)

        return mask

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_ids (Tensor of size (batch_size, seq_length)):
                Indices of input sequence tokens.
            padding_mask (Optional[Tensor] of size (batch_size, seq_length)):
                Boolean tensor: True for unpadded tokens, False for padded tokens.
        Returns:
            A tuple including
                pooled (Tensor): Normalized CLS embedding of shape
                    (batch_size, output_dim) (for use in contrastive loss).
                tokens (Tensor): Embeddings for all non-CLS tokens. Shape:
                    (batch_size, num_positions, output_dim).
        """

        # If using CLS embedding, drop the final token
        if self.embed_cls:
            if input_ids.shape[1] == self.num_positions:
                input_ids = input_ids[:, :-1]
            if padding_mask is not None and padding_mask.shape[1] == self.num_positions:
                padding_mask = padding_mask[:, :-1]

        target_shape = self.num_positions - 1 if self.embed_cls else self.num_positions
        assert (
            input_ids.shape[1] == target_shape
        ), f"{input_ids.shape} doesn't match ({target_shape},*)"

        embeddings = self.embeddings(input_ids)
        mask = self.build_mask(input_ids, padding_mask)
        decoder_out = self.transformer_decoder(embeddings, attention_mask=mask)
        hidden_states = decoder_out.last_hidden_state
        assert hidden_states is not None, "hidden states must not be None"
        if self.embed_cls:
            pooled, tokens = hidden_states[:, -1], hidden_states[:, :-1]
            if self.ln_final is not None:
                pooled = self.ln_final(pooled)
        else:
            hidden_states = self.ln_final(hidden_states)
            # Use argmax to get EOS embedding (assumes EOS token has highest value)
            pooled, tokens = (
                hidden_states[
                    torch.arange(hidden_states.shape[0]), input_ids.argmax(dim=-1)
                ],
                hidden_states,
            )

        if self.text_projection is not None:
            pooled = self.text_projection(pooled)

        return pooled, tokens
