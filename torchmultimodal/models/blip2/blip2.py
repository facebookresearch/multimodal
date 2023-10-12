# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Optional

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torchmultimodal.modules.layers.transformer import TransformerOutput


class Blip2Output(NamedTuple):
    """
    BLIP2 model output for loss computation.

    image_embeddings(Tensor): normalized image embeddings returned by the visual encoder
        with shape [bsz x seq_len x embed_dim].
    image_features(Tensor): Image features after qformer and projection (for stage 1 training)
        with shape [bsz, num_query_tokens, embed_dim]
    image_qformer_output(Tensor) : last hidden state for qformer output by given image input
    text_features(Optional[Tensor]): Text features after qformer and projection if text input is provided
        with shape [bsz, embed_dim]
    prediction_scores (Optional[Tensor]): computed for next word prediction
        with shape of [bsz, seq_len, vocab_size]
    """

    image_embeddings: Tensor
    image_features: Tensor
    image_qformer_output: Tensor
    text_features: Optional[Tensor] = None
    prediction_scores: Optional[Tensor] = None


class BLIP2(nn.Module):
    """
    BLIP2(https://arxiv.org/pdf/2301.12597.pdf) provides a pre-training strategy to bootstrap vision-language
    pre-training from frozen image encoders and frozen large language models(LLM). BLIP-2 bridges the modality gap
    and facilitates cross-modal alignment via Querying Transformer (Q-former). Q-former is a lightweight transformer
    which has a set of learnable query vectors to extract visual features from the frozen image encoder.

    Args:
        qformer(nn.Module): Querying Transformer (Q-former)
        visual_encoder(nn.Module): Frozen image encoder
        dim_q(int) : Dimension of query tensor, this value should be the same as dim_q in qformer.
        image_encoder_embedding_dim(int): Embedding dimension for image encoder,
            this value should be the same as dim_kv in qformer.
        freeze_visual_encoder(bool): Whether to freeze the visual encoder, default to True
        cross_attention_freq(int): Frequency of adding cross-attention block in Qformer, default to 2
        embedding_dim(int): Embedding dimension
        num_query_token(int): Number of query tokens in Qformer, default to 32
        init_query_tokens(bool): whether init query token params, default to True
        decoder_bos_token_id(Optional[int]): bos_token_id used in decoder, default to None
    """

    def __init__(
        self,
        qformer: nn.Module,
        vision_encoder: nn.Module,
        dim_q: int,
        image_encoder_embedding_dim: int,
        freeze_vision_encoder: bool = True,
        cross_attention_freq: int = 2,
        embedding_dim: int = 256,
        num_query_token: int = 32,
        init_query_tokens: bool = True,
        decoder_bos_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()

        self.qformer = qformer
        self.decoder_bos_token_id = decoder_bos_token_id
        self.dim_q = dim_q
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.dim_q))
        if init_query_tokens:
            self.query_tokens.data.normal_(mean=0.0, std=0.02)

        self.vision_proj = nn.Linear(self.dim_q, embedding_dim)
        self.text_proj = nn.Linear(self.dim_q, embedding_dim)
        self.ln_vision = nn.LayerNorm(image_encoder_embedding_dim)

    def forward(
        self,
        image: Tensor,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Blip2Output:
        """
        Args:
            image(Tensor): Image input tensor with shape [B, C, H, W]
            input_ids(Optional[Tensor]): Text input tensor with shape [bsz, seq_len]
            attention_mask(Optional[Tensor]): Attention mask tensor with shape [bsz, seq_len]

        Returns:
            return BLIP2 model output(Blip2Output).
        """
        vision_encoder_output = self.vision_encoder(image)
        if isinstance(vision_encoder_output, TransformerOutput):
            vision_encoder_output = vision_encoder_output.last_hidden_state
        assert vision_encoder_output is not None
        image_embeds = self.ln_vision(vision_encoder_output)
        # query tokens: [batch_size, num_query_token, encoder_hidden_size]
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.qformer.model(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            use_cache=True,
        )

        # image_feats: [batch_size, num_query_token, embedding_dim]
        image_feats = F.normalize(self.vision_proj(query_output[0]), dim=-1)

        text_feats: Optional[Tensor] = None
        prediction_scores: Optional[Tensor] = None
        if input_ids is not None:
            text_output = self.qformer.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            text_feats = F.normalize(self.text_proj(text_output[0][:, 0, :]), dim=-1)

            decoder_input_ids = input_ids.clone()
            if self.decoder_bos_token_id is not None:
                # pyre-ignore
                decoder_input_ids[:, 0] = self.decoder_bos_token_id

            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                input_ids.device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([query_atts, attention_mask], dim=1)

            # set use_cache = False since past_key_values should be cached in previous steps.
            prediction_scores = self.qformer(
                input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output[1],
                use_cache=False,
            )

        return Blip2Output(
            image_embeddings=image_embeds,
            image_features=image_feats,
            image_qformer_output=query_output[0],
            text_features=text_feats,
            prediction_scores=prediction_scores,
        )
