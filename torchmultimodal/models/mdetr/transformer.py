# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple, Optional

import torch
from torch import nn, Tensor
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.utils.common import get_clones


class MDETRTransformerOutput(NamedTuple):
    decoder_hidden_states: torch.Tensor
    text_memory: torch.Tensor


class MDETRTransformer(nn.Module):
    """
    Transformer class for MDETR model.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            num_encoder_layers (int): Number of layers in the encoder. Default: 6
            num_decoder_layers (int): Number of layers in the decoder. Default: 6
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU
            normalize_before (bool): Whether to do PreNorm. Default: False
            return_intermediate_dec (bool): Whether to return intermediate decoder outputs.
                Default: True

    Inputs: image_embeddings Tensor: The image input.
            image_mask (Tensor) The mask for the image sequence.
            query_embed (Tensor): Positional embeddings applied to Q
                cross-attention matrix in decoder.
            pos_embed (Tensor): Positional embeddings applied to Q and K
                self-attention matrices in decoder.
            text_memory (Tensor): Text input.
            text_attention_mask (Tensor): Attention mask for text input.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalize_before: bool = False,
        return_intermediate_dec: bool = True,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_final_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_final_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.d_model = d_model
        self._init_parameters()

    # Initialize all (non-normalization-layer) weights
    # Biases will be unaffected
    def _init_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        image_embeddings: Tensor,
        image_mask: Tensor,
        query_embed: Tensor,
        pos_embed: Tensor,
        text_memory: Tensor,
        text_attention_mask: Tensor,
    ) -> MDETRTransformerOutput:
        # flatten NxCxHxW to HWxNxC
        bs = image_embeddings.size(0)
        image_embeddings = image_embeddings.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # Object query embeddings for each sample in the batch
        # Size: (num_queries, batch_size, hidden_dim)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        image_mask = image_mask.flatten(1)

        tgt = torch.zeros_like(query_embed)

        # Concat on the sequence dimension
        mm_embeddings = torch.cat([image_embeddings, text_memory], dim=0)
        # For mask, sequence dimension is second
        image_mask = torch.cat([image_mask, text_attention_mask], dim=1)

        # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory)], dim=0)
        mm_memory = self.encoder(
            mm_embeddings, src_key_padding_mask=image_mask, pos=pos_embed
        )
        text_memory = mm_memory[-len(text_memory) :]
        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

        hs = self.decoder(
            tgt,
            mm_memory,
            memory_key_padding_mask=image_mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return MDETRTransformerOutput(
            decoder_hidden_states=hs.transpose(1, 2), text_memory=text_memory
        )


class TransformerEncoder(nn.Module):
    """
    A transformer encoder.

    Args:   encoder_layer (nn.Module): Module for an individual encoder layer.
            num_layers (int): Number of encoder layers.
            norm (Optional[nn.Module]): Normalization applied after last encoder layer.
                Default: None

    Inputs: src (Tensor): The sequence to the encoder layer.
            mask (Optional[Tensor]) The mask for the src sequence. Default: None
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch.
                Default: None
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:

        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    A transformer decoder.

    Args:   decoder_layer (nn.Module): Module for an individual decoder layer.
            num_layers (int): Number of decoder layers.
            norm (Optional[nn.Module]): Normalization applied after last decoder layer.
                Default: None
            return_intermediate (bool): Whether to return intermediate decoder outputs.
                Default: True

    Inputs: tgt (Tensor): The sequence to the decoder layer.
            memory (Tensor): The sequence from the last layer of the decoder.
            tgt_mask (Optional[Tensor]) The mask for the tgt sequence. Default: None
            memory_mask (Optional[Tensor]): The mask for the memory sequence.
                Default: None
            tgt_key_padding_mask (Optional[Tensor]): The mask for the tgt keys per batch.
                Default: None
            memory_key_padding_mask (Optional[Tensor]):
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
            query_pos (Optional[Tensor]): Positional embeddings applied to Q
                cross-attention matrix. Default: None
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if self.norm is not None:
            return self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    A single layer from a transformer encoder.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU
            normalize_before (bool): Whether to do PreNorm. Default: False

    Inputs: src (Tensor): The sequence to the encoder layer.
            src_mask (Optional[Tensor]) The mask for the src sequence. Default: None
            src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch.
                Default: None
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.mlp = MLP(d_model, d_model, [dim_feedforward], dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        q = k = self.with_pos_embed(x, pos)
        self_attention_outputs = self.self_attn(
            q, k, value=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        x = x + self.dropout1(self_attention_outputs)
        x = self.norm1(x)
        mlp_outputs = self.mlp(x)
        x = x + self.dropout2(mlp_outputs)
        x = self.norm2(x)
        return x

    def forward_pre(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        x = self.norm1(x)
        q = k = self.with_pos_embed(x, pos)
        self_attention_outputs = self.self_attn(
            q, k, value=x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        x = x + self.dropout1(self_attention_outputs)
        x = self.norm2(x)
        mlp_outputs = self.mlp(x)
        x = x + self.dropout2(mlp_outputs)
        return x

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    A single layer from a transformer decoder.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU

    Inputs: tgt (Tensor): The sequence to the decoder layer.
            memory (Tensor): The sequence from the last layer of the decoder.
            tgt_mask (Optional[Tensor]) The mask for the tgt sequence. Default: None
            memory_mask (Optional[Tensor]): The mask for the memory sequence.
                Default: None
            tgt_key_padding_mask (Optional[Tensor]): The mask for the tgt keys per batch.
                Default: None
            memory_key_padding_mask (Optional[Tensor]):
            pos (Optional[Tensor]): Positional embeddings applied to Q and K
                self-attention matrices. Default: None
            query_pos (Optional[Tensor]): Positional embeddings applied to Q
                cross-attention matrix. Default: None
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout
        )
        self.mlp = MLP(d_model, d_model, [dim_feedforward], dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        x = tgt
        q = k = self.with_pos_embed(x, query_pos)

        # Self attention
        self_attention_outputs = self.self_attn(
            q, k, value=x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        x = x + self.dropout1(self_attention_outputs)
        x = self.norm1(x)

        # Cross attention to image
        cross_attention_outputs = self.cross_attn_image(
            query=self.with_pos_embed(x, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        x = x + self.dropout3(cross_attention_outputs)
        x = self.norm3(x)

        # FFN
        mlp_outputs = self.mlp(x)
        x = x + self.dropout4(mlp_outputs)
        x = self.norm4(x)
        return x


def mdetr_transformer(
    d_model: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    return_intermediate_dec: bool = True,
) -> MDETRTransformer:
    return MDETRTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        return_intermediate_dec=return_intermediate_dec,
    )
