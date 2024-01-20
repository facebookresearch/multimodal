# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.models.coca.multimodal_decoder import CoCaMultimodalDecoder
from torchmultimodal.models.coca.text_decoder import CoCaTextDecoder
from torchmultimodal.modules.encoders.vision_transformer import vision_transformer
from torchmultimodal.modules.layers.attention_pooler import (
    AttentionPooler,
    CascadedAttentionPooler,
)
from torchmultimodal.modules.layers.transformer import TransformerOutput
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)


class MultimodalOutput(NamedTuple):
    image_pooled_output: Tensor
    text_pooled_output: Tensor
    multimodal_embeddings: Tensor
    multimodal_pooled_embeddings: Optional[Tensor] = None


class CoCaModel(nn.Module):
    """
    CoCa model class containing vision encoder, text decoder, and multimodal decoder.
    Reference: https://arxiv.org/abs/2205.01917
    Args:
        vision_encoder (nn.Module): Instantiated vision encoder. Should return either
            TransformerOutput or Tensor.
        text_decoder (CoCaTextDecoder): Instantiated CoCaTextDecoder returning a
            Tuple[Tensor, Tensor], where the first element is the normalized CLS
            embedding, and the second element is the full set of token embeddings.
        multimodal_decoder (nn.Module): Instantiated CoCaMultimodalDecoder returning a
            Tensor of multimodal embeddings.
        vision_pooler (nn.Module): Pooler for vision outputs (see e.g. AttentionPooler).
        vision_proj (nn.Module): Projection layer for vision encoder. Note that the
            projections for the text_decoder and multimodal_decoder are handled inside
            the CoCaTextDecoder and CoCaMultimodalDecoder classes, respectively, but
            for vision we apply attentional pooling first so the vision projection
            is separated from the vision_encoder class.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,  # e.g. ViT
        text_decoder: CoCaTextDecoder,
        multimodal_decoder: CoCaMultimodalDecoder,
        vision_pooler: nn.Module,
        vision_proj: nn.Module,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.multimodal_decoder = multimodal_decoder
        self.vision_pooler = vision_pooler
        self.vision_proj = vision_proj

    def forward(
        self, images: Tensor, texts: Tensor, text_padding_mask: Optional[Tensor] = None
    ) -> MultimodalOutput:
        """
        Args:
            images (Tensor): Tensor of size (bsz, c, h, w) containing image pixels.
            texts (Tensor): Tensor of size (bsz, seq_len) containing text tokens.
            text_padding_mask (Optional[Tensor]): Boolean mask indicating padded tokens.
                True for unpadded tokens, False for padded tokens. Default: None
        Returns:
            MultimodalOutput containing pooled image embeddings, text embeddings,
                and multimodal embeddings.
        """

        # Image encoder
        vision_encoder_outs = self.vision_encoder(images)

        # Extract image embeddings
        if isinstance(vision_encoder_outs, TransformerOutput):
            image_embeddings = vision_encoder_outs.last_hidden_state
        elif isinstance(vision_encoder_outs, tuple):
            vision_encoder_outs = vision_encoder_outs[0]
            assert isinstance(vision_encoder_outs, Tensor)
            image_embeddings = vision_encoder_outs
        else:
            assert isinstance(vision_encoder_outs, Tensor)
            image_embeddings = vision_encoder_outs
        assert isinstance(image_embeddings, Tensor), "Image embeddings must be Tensor"

        pooled_outputs = self.vision_pooler(image_embeddings)
        if torch.jit.isinstance(pooled_outputs, List[Tensor]):
            assert len(pooled_outputs) == 2
            captioning_image_embeddings, contrastive_image_embeddings = pooled_outputs
        else:
            assert isinstance(
                pooled_outputs, Tensor
            ), "Pooled image embeddings must be Tensor"
            # For parallel pooler arch of CoCa, we use a single pooler and split
            # the outputs for contrastive and captioning tasks
            contrastive_image_embeddings, captioning_image_embeddings = (
                pooled_outputs[:, 0],
                pooled_outputs[:, 1:],
            )
        contrastive_image_embeddings = self.vision_proj(contrastive_image_embeddings)
        contrastive_image_embeddings = F.normalize(contrastive_image_embeddings, dim=-1)

        # Text decoder
        pooled_text_embeddings, text_tokens = self.text_decoder(
            texts, text_padding_mask
        )
        contrastive_text_embeddings = F.normalize(pooled_text_embeddings, dim=-1)

        # Multimodal decoder
        multimodal_embeddings = self.multimodal_decoder(
            text_tokens, captioning_image_embeddings
        )

        return MultimodalOutput(
            contrastive_image_embeddings,
            contrastive_text_embeddings,
            multimodal_embeddings,
        )


def coca_vit(
    *,
    # Required vision args
    vision_patch_size: int,
    vision_dim_feedforward: int,
    vision_n_layer: int,
    vision_n_head: int,
    # Required text args
    vocab_size: int,
    num_text_positions: int,
    text_hidden_dim: int,
    text_n_layer: int,
    text_n_head: int,
    text_dim_feedforward: int,
    text_output_dim: int,
    # Required fusion args
    fusion_n_layer: int,
    fusion_n_head: int,
    fusion_dim_feedforward: int,
    # Required attention pooler args
    pooler_input_embed_dim: int,
    pooler_output_embed_dim: int,
    pooler_n_head: int,
    # Optional vision args
    image_size: Union[int, Tuple[int, int]] = 224,
    num_channels: int = 3,
    vision_activation: Callable[..., nn.Module] = nn.GELU,
    vision_transformer_dropout: float = 0.0,
    patch_embed_dropout_prob: float = 0.0,
    vision_layer_norm_eps: float = 1e-5,
    vision_final_layer_norm_eps: Optional[float] = None,
    vision_norm_first: bool = True,
    vision_include_cls_embed: bool = False,  # This is different from ViT default
    vision_drop_path_rate: Optional[float] = None,
    vision_patch_drop_rate: Optional[Union[float, Tuple[float, float]]] = None,
    # Optional text args
    pad_idx: Optional[int] = 0,
    text_embed_cls: bool = True,
    text_dropout: float = 0.0,
    text_activation: Callable[..., nn.Module] = nn.GELU,
    text_layer_norm_eps: float = 1e-5,
    text_norm_first: bool = True,
    text_final_layer_norm_eps: Optional[float] = 1e-5,
    # Optional fusion args
    fusion_dropout: float = 0.0,
    fusion_activation: Callable[..., nn.Module] = nn.GELU,
    fusion_layer_norm_eps: float = 1e-5,
    fusion_norm_first: bool = True,
    fusion_final_layer_norm_eps: Optional[float] = 1e-5,
    multimodal_output_projection_dim: Optional[int] = None,
    # Optional attention pooler args
    cascaded_pooler: bool = True,
    pooler_n_queries: int = 256,
    pooler_layer_norm_eps: float = 1e-5,
) -> CoCaModel:
    """
    Args:
        vision_patch_size (Union[int, Tuple[int, int]]): ViT patch size
        vision_dim_feedforward (int): Dimension of FFN for ViT encoder.
        vision_n_layer (int): Number of layers in ViT encoder
        vision_n_head (int): Number of heads in ViT encoder.
        vocab_size (int): Text vocab size.
        num_text_positions (int): Number of positions for text tokens.
        text_hidden_dim (int): Embedding dimension in text transformer.
        text_n_layer (int): Number of layers in text transformer.
        text_n_head (int): Number of heads in text transformer.
        text_dim_feedforward (int): Dimension of FFN for text transformer.
        text_output_dim (int): Output dimension of text decoder.
        fusion_n_layer (int): Number of layers in multimodal transformer.
        fusion_n_head (int): Number of heads in multimodal transformer.
        fusion_dim_feedforward (int): Dimension of FFN for multimodal transformer.
        pooler_input_embed_dim (int): Input dimension for attention pooler.
        pooler_output_embed_dim (int): Output dimension for attention pooler.
        pooler_n_head (int): Number of heads in attention pooler.
        image_size (Union[int, Tuple[int, int]]): Size of input image. Default: 224
        num_channels (int): Number of channels of image. Default: 3
        vision_activation (Callable[..., nn.Module]): ViT activation function.
            Default: GELU
        vision_transformer_dropout (float): ViT encoder dropout rate. Default: 0.0
        patch_embed_dropout_prob (float): Image patch embedding dropout rate.
            Default: 0.0
        vision_layer_norm_eps (float): LN epsilon in ViT encoder. Default: 1e-5
        vision_final_layer_norm_eps (float): Final LN epsilon for ViT.
            Default: 0.0 (no final LN)
        vision_norm_first (bool): Whether to use pre-norm ViT layers. Default: True
        vision_include_cls_embed (bool): Whether to include cls as an embedding
            Default: False (to match open_clip implementation)
        vision_drop_path_rate (Optional[float]): Stochastic drop path rate in ViT.
            Default: None (no drop path)
        vision_patch_drop_rate (Optional[Union[float, Tuple[float, float]]]): Rate
        for masking patches prior to ViT encoder. Default: None (no masking)
        pad_idx (int): Padding index of text. Default: 0
        text_embed_cls (bool): Whether to replace the final position of text with cls
            embedding. Default: True
        text_dropout (float): Dropout rate for text transformer. Default: 0.0
        text_activation (Callable[..., nn.Module]): Text transformer activation
            function. Default: GELU
        text_layer_norm_eps (float): LN epsilon in text transformer. Default: 1e-5
        text_norm_first (bool): Whether to use pre-norm layers in text decoder.
            Default: True
        text_final_layer_norm_eps (float): Final LN epsilon for text decoder.
            Default: 0.0 (no final LN)
        fusion_dropout (float): Dropout rate for multimodal transformer. Default: 0.0
        fusion_activation (Callable[..., nn.Module]): Activation function for
            multimodal transformer. Default: GELU
        fusion_layer_norm_eps (float): LN epsilon in multimodal transformer.
            Default: 1e-5
        fusion_norm_first (bool): Whether to use pre-norm layers in multimodal decoder.
            Default: True
        fusion_final_layer_norm_eps (float): Final LN epsilon for multimodal decoder.
            Default: 0.0 (no final LN)
        multimodal_output_projection_dim (Optional[int]): Output dimension of
            multimodal projection. If None, no projection will be applied to
            multimodal embeddings. Default: None
        cascaded_pooler (bool): Whether to cascade (stack) contrastive and captioning
            attention poolers or parallelize them. Default: True
        pooler_n_queries (int): Number of queries in captioning attention pooler.
            Contrastive attention pooler always has one query. For parallel pooling,
            the attention pooler will have a single pooler using n_queries+1 queries
            with the first position for contrastive embeddings. For cascaded pooling,
            the first pooler is the captioning pooler with pooler_n_queries queries
            and the second is the contrastive pooler with one query. Default: 256
        pooler_layer_norm_eps (float): LN epsilon in attention pooler. Default: 1e-5
    """
    attention_pooler: nn.Module
    if cascaded_pooler:
        captioning_pooler = AttentionPooler(
            input_embed_dim=pooler_input_embed_dim,
            output_embed_dim=pooler_output_embed_dim,
            n_head=pooler_n_head,
            n_queries=pooler_n_queries,
            layer_norm_eps=pooler_layer_norm_eps,
        )
        contrastive_pooler = AttentionPooler(
            input_embed_dim=pooler_output_embed_dim,
            output_embed_dim=pooler_output_embed_dim,
            n_head=pooler_n_head,
            n_queries=1,
            layer_norm_eps=pooler_layer_norm_eps,
        )
        attention_pooler = CascadedAttentionPooler(
            [captioning_pooler, contrastive_pooler]
        )
    else:
        attention_pooler = AttentionPooler(
            input_embed_dim=pooler_input_embed_dim,
            output_embed_dim=pooler_output_embed_dim,
            n_head=pooler_n_head,
            n_queries=pooler_n_queries + 1,
            layer_norm_eps=pooler_layer_norm_eps,
        )

    vision_proj = nn.Linear(
        pooler_output_embed_dim, pooler_output_embed_dim, bias=False
    )
    nn.init.normal_(vision_proj.weight, std=pooler_input_embed_dim**-0.5)

    vision_encoder = vision_transformer(
        patch_size=vision_patch_size,
        hidden_dim=pooler_input_embed_dim,
        dim_feedforward=vision_dim_feedforward,
        n_layer=vision_n_layer,
        n_head=vision_n_head,
        image_size=image_size,
        num_channels=num_channels,
        activation=vision_activation,
        transformer_dropout=vision_transformer_dropout,
        patch_embed_dropout_prob=patch_embed_dropout_prob,
        layer_norm_eps=vision_layer_norm_eps,
        final_layer_norm_eps=vision_final_layer_norm_eps,
        norm_first=vision_norm_first,
        include_cls_embed=vision_include_cls_embed,
        drop_path_rate=vision_drop_path_rate,
        patch_drop_rate=vision_patch_drop_rate,
    )

    text_decoder = CoCaTextDecoder(
        vocab_size=vocab_size,
        num_positions=num_text_positions,
        embedding_dim=text_hidden_dim,
        n_layer=text_n_layer,
        n_head=text_n_head,
        dim_feedforward=text_dim_feedforward,
        output_dim=text_output_dim,
        pad_idx=pad_idx,
        embed_cls=text_embed_cls,
        dropout=text_dropout,
        activation=text_activation,
        layer_norm_eps=text_layer_norm_eps,
        norm_first=text_norm_first,
        final_layer_norm_eps=text_final_layer_norm_eps,
    )

    mm_input_seq_len = num_text_positions - 1 if text_embed_cls else num_text_positions

    multimodal_decoder = CoCaMultimodalDecoder(
        input_seq_len=mm_input_seq_len,
        text_embedding_dim=pooler_output_embed_dim,
        n_layer=fusion_n_layer,
        n_head=fusion_n_head,
        dim_feedforward=fusion_dim_feedforward,
        output_dim=multimodal_output_projection_dim,
        dropout=fusion_dropout,
        activation=fusion_activation,
        layer_norm_eps=fusion_layer_norm_eps,
        norm_first=fusion_norm_first,
        final_layer_norm_eps=fusion_final_layer_norm_eps,
    )

    return CoCaModel(
        vision_encoder=vision_encoder,
        text_decoder=text_decoder,
        multimodal_decoder=multimodal_decoder,
        vision_proj=vision_proj,
        vision_pooler=attention_pooler,
    )


def coca_vit_b_32() -> CoCaModel:
    return coca_vit(
        vision_patch_size=32,
        vision_n_layer=12,
        vision_n_head=12,
        vision_dim_feedforward=3072,
        vision_include_cls_embed=False,
        vocab_size=49408,
        num_text_positions=77,
        text_hidden_dim=512,
        text_n_layer=12,
        text_n_head=8,
        text_dim_feedforward=2048,
        text_output_dim=512,
        fusion_n_layer=12,
        fusion_n_head=8,
        fusion_dim_feedforward=2048,
        multimodal_output_projection_dim=49408,
        pooler_input_embed_dim=768,
        pooler_output_embed_dim=512,
        pooler_n_head=8,
        cascaded_pooler=True,
    )


def coca_vit_l_14() -> CoCaModel:
    return coca_vit(
        vision_patch_size=14,
        vision_n_layer=24,
        vision_n_head=16,
        vision_dim_feedforward=4096,
        vision_include_cls_embed=False,
        vocab_size=49408,
        num_text_positions=77,
        text_hidden_dim=768,
        text_n_layer=12,
        text_n_head=12,
        text_dim_feedforward=3072,
        text_output_dim=768,
        fusion_n_layer=12,
        fusion_n_head=12,
        fusion_dim_feedforward=3072,
        multimodal_output_projection_dim=49408,
        pooler_input_embed_dim=1024,
        pooler_output_embed_dim=768,
        pooler_n_head=8,
        cascaded_pooler=True,
    )


class CoCaForPretraining(nn.Module):
    """
    CoCa pretraining model class.
    Ties CoCa model to captioning and contrastive losses.
    Args:
        model (CoCaModel): Instantiated CoCa model.
        pad_idx (int): Index of padding tokens (used to filter captioning
        loss indices). Default: 0
        contrastive_logit_scale_min (Optional[float]): Min clamp value for contrastive
            temperature. Default: 0.0
        contrastive_logit_scale_max (Optional[float]): Max clamp value for contrastive
            temperature. Default: log(100)
    """

    def __init__(
        self,
        model: CoCaModel,
        pad_idx: int = 0,
        contrastive_logit_scale_min: Optional[float] = math.log(1.0),
        contrastive_logit_scale_max: Optional[float] = math.log(100.0),
    ):
        super().__init__()
        self.model = model
        self.contrastive_loss = ContrastiveLossWithTemperature(
            logit_scale_min=contrastive_logit_scale_min,
            logit_scale_max=contrastive_logit_scale_max,
        )
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(
        self, images: Tensor, texts: Tensor, text_padding_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        Args:
            images (Tensor): Tensor of size (bsz, c, h, w) containing image pixels.
            texts (Tensor): Tensor of size (bsz, seq_len) containing text tokens.
            text_padding_mask (Optional[Tensor]): Boolean mask indicating padded tokens.
                True for unpadded tokens, False for padded tokens. Default: None
        Returns:
            Dict[str, Tensor]: Dict containing contrastive and captioning losses with
                respective keys 'contrastive' and 'captioning'.
        """
        model_outs = self.model(images, texts, text_padding_mask)
        captioning_labels = texts[:, 1:].contiguous()
        contrastive_loss = self.contrastive_loss(
            model_outs.image_pooled_output, model_outs.text_pooled_output
        )

        vocab_size = model_outs.multimodal_embeddings.shape[-1]
        captioning_loss = self.caption_loss(
            model_outs.multimodal_embeddings.view(-1, vocab_size),
            captioning_labels.view(-1),
        )
        return {"contrastive": contrastive_loss, "captioning": captioning_loss}


def coca_for_pretraining(pad_idx: int = 0, **kwargs: Any) -> CoCaForPretraining:
    model = coca_vit(**kwargs)
    return CoCaForPretraining(model, pad_idx=pad_idx)


default_coca_cls_pooler = partial(torch.select, dim=1, index=-1)


class CoCaModelWithHeads(nn.Module):
    """
    CoCa model with heads.
    Args:
        model (CoCaModel): Instantiated CoCa model.
        heads (nn.ModuleDict): Dictionary of heads, taking either unimodal or
            multimodal embeddings as inputs
        pad_idx (int): Index of padding tokens (used to filter captioning
        loss indices). Default: 0
        pooler (Callable): how to extract the the multimodal embeddings. some examples
            [default] partial(torch.select, dim=1, index=-1)
            partial(torch.mean, dim=1)
            partial(torch.max, dim=1)
            torchmultimodal.fb.modules.layers.attention_pooler.AttentionPooler
    """

    def __init__(
        self,
        model: CoCaModel,
        heads: nn.ModuleDict,
        pad_idx: int = 0,
        pooler: Callable = default_coca_cls_pooler,
    ):
        super().__init__()
        self.model = model
        self.heads = heads
        self.pooler = pooler

    def forward(
        self, images: Tensor, texts: Tensor, text_padding_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:

        model_out = self.model(images, texts, text_padding_mask)
        mm_out = model_out.multimodal_embeddings

        bsz = mm_out.shape[0]
        # reshape in the case of attention pooler
        pooled_output = self.pooler(mm_out).view((bsz, -1))

        # run the heads
        head_outputs = {}
        for k, head in self.heads.items():
            head_outputs[k] = head(pooled_output)

        return head_outputs
