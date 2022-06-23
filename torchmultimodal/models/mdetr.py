# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchmultimodal.modules.encoders.mdetr_image_encoder import (
    mdetr_resnet101_backbone,
    PositionEmbedding2D,
)
from torchmultimodal.modules.encoders.mdetr_text_encoder import (
    mdetr_roberta_text_encoder,
)
from torchmultimodal.modules.layers.mlp import MLP
from torchvision.models import resnet101


class MDETR(nn.Module):
    """
    MDETR (https://arxiv.org/abs/2104.12763) is a modulated detection model
    used to detect objects in an image conditioned on text or captions.
    This class contains the entire MDETR architecture, including the
    image backbone, text encoder, and multimodal transformer. (Note that the
    matcher and losses are provided elsewhere.)

    Args:   image_backbone (nn.Module): Torch module of the backbone to be used.
                See mdetr_image_encoder.py.
            text_encoder (nn.Module): Torch module of the text encoder to be used.
                See mdetr_text_encoder.py.
            transformer (nn.Module): The multimodal transformer module. See the
                Transformer class in this file.
            pos_embed (nn.Module): Module for positional embedding of images.
            resizer (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            input_proj (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.


    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes)
            text (List[Tensor]): A list of Tensors of tokenized texts (possibly of different lengths).

    Returns:
        A dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
    """

    def __init__(
        self,
        image_backbone: nn.Module,
        text_encoder: nn.Module,
        transformer: nn.Module,
        pos_embed: nn.Module,
        resizer: nn.Module,
        input_proj: nn.Module,
        query_embed: nn.Module,
        bbox_embed: nn.Module,
        class_embed: nn.Module,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        self.resizer = resizer
        self.transformer = transformer
        self.pos_embed = pos_embed
        self.input_proj = input_proj
        self.query_embed = query_embed
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed

    def _pad_images(self, images: List[Tensor]) -> Tuple[Tensor, Tensor]:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        b, _, h, w = batch_shape

        dtype = images[0].dtype
        device = images[0].device
        padded_images = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(images, padded_images, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        return padded_images, mask

    def _pad_text(
        self, text: List[Tensor], padding_idx: int = 1
    ) -> Tuple[Tensor, Tensor]:
        padded_text = nn.utils.rnn.pad_sequence(
            text, batch_first=True, padding_value=padding_idx
        )
        mask = padded_text == padding_idx
        return padded_text, mask

    def forward(self, images: List[Tensor], text: List[Tensor]):

        images, image_mask = self._pad_images(images)
        text, text_attention_mask = self._pad_text(text)
        encoded_text = self.text_encoder(text, text_attention_mask)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.transpose(0, 1)

        src, mask = self.image_backbone(images, image_mask)
        pos = self.pos_embed(mask).to(src.dtype)
        query_embed = self.query_embed.weight

        text_memory_resized = self.resizer(text_memory)

        hs = self.transformer(
            self.input_proj(src),
            mask,
            query_embed,
            pos,
            text_memory=text_memory_resized,
            img_memory=None,
            text_attention_mask=text_attention_mask,
        )
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }

        return out


class Transformer(nn.Module):
    """
    MDETR (https://arxiv.org/abs/2104.12763) is a modulated detection model
    used to detect objects in an image conditioned on text or captions.
    This class contains the entire MDETR architecture, including the
    image backbone, text encoder, and multimodal transformer. (Note that the
    matcher and losses are provided elsewhere.)

    Args:   image_backbone (nn.Module): Torch module of the backbone to be used.
                See mdetr_image_encoder.py.
            text_encoder (nn.Module): Torch module of the text encoder to be used.
                See mdetr_text_encoder.py.
            transformer (nn.Module): The multimodal transformer module. See the
                Transformer class in this file.
            pos_embed (nn.Module): Module for positional embedding of images.
            resizer (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            input_proj (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.


    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes)
            text (List[Tensor]): A list of Tensors of tokenized texts (possibly of different lengths).
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., Tensor] = nn.functional.relu,
        normalize_before: bool = False,
        return_intermediate_dec: bool = False,
        pass_pos_and_query: bool = True,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
    ):
        # flatten NxCxHxW to HWxNxC
        bs, _, _, _ = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        if self.pass_pos_and_query:
            tgt = torch.zeros_like(query_embed)
        else:
            src, tgt, query_embed, pos_embed = (
                src + 0.1 * pos_embed,
                query_embed,
                None,
                None,
            )

        # Concat on the sequence dimension
        src = torch.cat([src, text_memory], dim=0)
        # For mask, sequence dimension is second
        mask = torch.cat([mask, text_attention_mask], dim=1)

        # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory)], dim=0)
        img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        text_memory = img_memory[-len(text_memory) :]
        assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

        if not self.pass_pos_and_query:
            pos_embed = None

        assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

        hs = self.decoder(
            tgt,
            img_memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2)


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
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

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
                Default: False

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
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
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
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    A single layer from a transformer decoder.

    Args:   d_model (int): Number of features in the input.
            nhead (int): Number of heads in multi-head attention.
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., Tensor]): The activation function of the
                intermediate layer. Default: relu
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
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., Tensor] = nn.functional.relu,
        normalize_before: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    A single layer from a transformer decoder.

    Args:   d_model (int): Number of features in the input.
            nhead (int): Number of heads in multi-head attention.
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., Tensor]): The activation function of the
                intermediate layer. Default: relu

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
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[..., Tensor] = nn.functional.relu,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = activation

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).

    Args:   input_feat_size (int): Dimension of input features.
            output_feat_size (int): Dimension of output features.
            dropout (float): Dropout probability for final features. Default: 0.1
            do_ln (bool): Whether to perform layer normalization after the linear layer.

    Inputs: encoder_features (Tensor): Features to be resized.
    """

    def __init__(
        self,
        input_feat_size: int,
        output_feat_size: int,
        dropout: float = 0.1,
        do_ln: bool = True,
    ):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: Tensor):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for i in range(N)])


def mdetr_transformer(
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    return_intermediate_dec: bool = True,
    pass_pos_and_query: bool = True,
) -> Transformer:
    return Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        return_intermediate_dec=return_intermediate_dec,
        pass_pos_and_query=pass_pos_and_query,
    )


def mdetr_resnet101(
    num_queries: int = 100,
    num_classes: int = 255,
    embedding_dim: int = 768,
    transformer_d_model: int = 256,
    transformer_nhead: int = 8,
    transformer_encoder_layers: int = 6,
    transformer_decoder_layers: int = 6,
    transformer_dim_feedforward: int = 2048,
    transformer_dropout: float = 0.1,
    return_intermediate_dec: bool = True,
    transformer_pass_pos_and_query: bool = True,
) -> MDETR:
    image_backbone = resnet101()
    image_backbone = mdetr_resnet101_backbone()
    pos_embed = PositionEmbedding2D(128, scale=2 * math.pi)
    image_backbone.num_channels = 2048
    text_encoder = mdetr_roberta_text_encoder()
    if embedding_dim is None:
        embedding_dim = text_encoder.embedding_dim
    transformer = mdetr_transformer(
        transformer_d_model,
        transformer_nhead,
        transformer_encoder_layers,
        transformer_decoder_layers,
        transformer_dim_feedforward,
        transformer_dropout,
        return_intermediate_dec,
        transformer_pass_pos_and_query,
    )
    hidden_dim = transformer_d_model
    resizer = FeatureResizer(embedding_dim, hidden_dim)
    input_proj = nn.Conv2d(image_backbone.num_channels, hidden_dim, kernel_size=1)
    query_embed = nn.Embedding(num_queries, hidden_dim)
    bbox_embed = MLP(hidden_dim, 4, [hidden_dim] * 2, dropout=0.0)
    class_embed = nn.Linear(hidden_dim, num_classes + 1)
    mdetr = MDETR(
        image_backbone,
        text_encoder,
        transformer,
        pos_embed,
        resizer,
        input_proj,
        query_embed,
        bbox_embed,
        class_embed,
    )
    return mdetr
