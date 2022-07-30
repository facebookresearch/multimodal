# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
from typing import Callable, List, NamedTuple, Optional, Tuple

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


class MDETRTransformerOutput(NamedTuple):
    decoder_hidden_states: torch.Tensor
    text_memory: torch.Tensor


class MDETRModelOutput(NamedTuple):
    transformer_output: MDETRTransformerOutput
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    extra_embeddings: Optional[torch.Tensor]


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
            text_projection (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            image_projection (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.
            extra_query_embeddings (Optional[nn.Embedding]): Additional query embeddings,
                as used in e.g. VQA. Default: None

    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes).
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
        text_projection: nn.Module,
        image_projection: nn.Module,
        query_embed: nn.Embedding,
        bbox_embed: nn.Module,
        class_embed: nn.Module,
        extra_query_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.transformer = transformer
        self.pos_embed = pos_embed
        self.image_projection = image_projection
        self.query_embed = query_embed
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed
        self.extra_query_embeddings = extra_query_embeddings

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

    def forward(self, images: List[Tensor], text: List[Tensor]) -> MDETRModelOutput:

        images, image_mask = self._pad_images(images)
        text, text_attention_mask = self._pad_text(text)
        encoded_text = self.text_encoder(text, text_attention_mask)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.transpose(0, 1)

        image_embeddings, image_mask = self.image_backbone(images, image_mask)
        pos = self.pos_embed(image_mask).to(image_embeddings.dtype)
        query_embed = self.query_embed.weight

        # If extra embeddings are provided for VQA, we concatenate them with
        # the other query embeddings prior to the transformer
        if self.extra_query_embeddings is not None:
            n_extra_embeddings = self.extra_query_embeddings.num_embeddings
            query_embed = torch.cat([query_embed, self.extra_query_embeddings.weight])

        text_memory_resized = self.text_projection(text_memory)

        transformer_output = self.transformer(
            self.image_projection(image_embeddings),
            image_mask,
            query_embed,
            pos,
            text_memory=text_memory_resized,
            text_attention_mask=text_attention_mask,
        )

        # Detach the extra embeddings from the hidden states returned by the decoder
        if self.extra_query_embeddings is not None:
            extra_embeddings = transformer_output.decoder_hidden_states[
                0, :, -n_extra_embeddings:
            ]
            decoder_hidden_states_truncated = transformer_output.decoder_hidden_states[
                :, :, :-n_extra_embeddings
            ]
            transformer_output = transformer_output._replace(
                decoder_hidden_states=decoder_hidden_states_truncated
            )
        else:
            extra_embeddings = None
        final_hidden_state = transformer_output.decoder_hidden_states[-1]
        outputs_class = self.class_embed(final_hidden_state)
        outputs_coord = self.bbox_embed(final_hidden_state).sigmoid()

        return MDETRModelOutput(
            transformer_output, outputs_class, outputs_coord, extra_embeddings
        )


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
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12) if do_ln else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: Tensor) -> Tensor:
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([deepcopy(module) for i in range(n)])


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


def mdetr_resnet101(
    num_queries: int = 100,
    num_classes: int = 255,
    embedding_dim: int = 768,
    transformer_d_model: int = 256,
    transformer_num_heads: int = 8,
    transformer_encoder_layers: int = 6,
    transformer_decoder_layers: int = 6,
    transformer_dim_feedforward: int = 2048,
    transformer_dropout: float = 0.1,
    return_intermediate_dec: bool = True,
    num_extra_query_embeddings: Optional[int] = None,
) -> MDETR:
    image_backbone = mdetr_resnet101_backbone()
    pos_embed = PositionEmbedding2D(128, scale=2 * math.pi)
    # Size of layer4 output in ResNet101. See
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204
    image_backbone_num_channels = 2048
    text_encoder = mdetr_roberta_text_encoder()
    transformer = mdetr_transformer(
        transformer_d_model,
        transformer_num_heads,
        transformer_encoder_layers,
        transformer_decoder_layers,
        transformer_dim_feedforward,
        transformer_dropout,
        return_intermediate_dec,
    )
    hidden_dim = transformer_d_model
    text_projection = FeatureResizer(embedding_dim, hidden_dim)
    image_projection = nn.Conv2d(image_backbone_num_channels, hidden_dim, kernel_size=1)
    query_embed = nn.Embedding(num_queries, hidden_dim)
    # 4 gives the number of coordinates that represent the bounding box
    bbox_embed = MLP(hidden_dim, 4, [hidden_dim] * 2, dropout=0.0)
    # The + 1 here corresponds to the "no class" label
    class_embed = nn.Linear(hidden_dim, num_classes + 1)
    if num_extra_query_embeddings is not None:
        extra_query_embeddings = nn.Embedding(num_extra_query_embeddings, hidden_dim)
    else:
        extra_query_embeddings = None

    mdetr = MDETR(
        image_backbone,
        text_encoder,
        transformer,
        pos_embed,
        text_projection,
        image_projection,
        query_embed,
        bbox_embed,
        class_embed,
        extra_query_embeddings,
    )
    return mdetr
