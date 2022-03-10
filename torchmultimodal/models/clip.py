# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torchmultimodal.modules.encoders.clip_resnet_encoder import ResNetForCLIP
from torchtext.models.roberta.modules import TransformerEncoder
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.prototype.models.vision_transformer import VisionTransformer


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder class. Should be instantiated and passed to CLIPModel

    As in CLIP, the text encoder follows a Transformer architecture.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders.
        context_length (int): Maximum sequence length for Transforer.
        vocab_size (int): Vocab size.
        width (int): Embedding dimension for Transformer encoder.
        heads (int): Number of heads in Transformer encoder.
        layers (int): Number of layers in Transformer encoder.
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs: text (Tensor): Tensor containing text features.
    """

    TOKEN_EMBEDDING_INIT_STD = 0.02
    POS_EMBEDDING_INIT_STD = 0.01

    def __init__(
        self,
        embedding_dim: int = 512,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        use_clip_init: bool = True,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embedding_dim=width,
            padding_idx=-1,
            max_seq_len=context_length,
            num_encoder_layers=layers,
            num_attention_heads=heads,
            dropout=0.0,
            normalize_before=True,
        )

        self.width = width
        self.context_length = context_length

        self.ln_final = nn.LayerNorm(width)
        self.projection = nn.Linear(width, embedding_dim, bias=False)

        if use_clip_init:
            self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize token and positional embeddings
        nn.init.normal_(
            self.encoder.token_embedding.weight, std=self.TOKEN_EMBEDDING_INIT_STD
        )
        nn.init.normal_(
            self.encoder.positional_embedding.embedding.weight,
            std=self.POS_EMBEDDING_INIT_STD,
        )

        proj_std = (self.width ** -0.5) * ((2 * len(self.encoder.layers)) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for layer in self.encoder.layers:
            nn.init.normal_(layer.attention.input_projection.weight, std=attn_std)
            nn.init.normal_(layer.attention.output_projection.weight, std=proj_std)
            # c_fc in CLIP corresponds to the first residual MLP layer
            nn.init.normal_(layer.residual_mlp.mlp[0].weight, std=fc_std)
            # c_proj in CLIP corresponds to the last residual MLP layer
            nn.init.normal_(layer.residual_mlp.mlp[-2].weight, std=proj_std)

        # Initialize projection
        nn.init.normal_(self.projection.weight, std=self.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.full((self.context_length, self.context_length), True).triu(1)
        return mask.to(dtype=bool)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(text, attn_mask=self.build_attention_mask())
        # [n_ctx, bs, transformer.width] -> [bs, n_ctx, transformer.width]
        embeddings = torch.permute(embeddings, (1, 0, 2))
        embeddings = self.ln_final(embeddings)
        # take features from the eot embedding (the highest number in each sequence)
        embeddings = self.projection(
            embeddings[torch.arange(embeddings.shape[0]), text.argmax(dim=-1)]
        )
        # embeddings now has size [bs, embedding_dim]

        return embeddings


class CLIPModel(nn.Module):
    """CLIP is a model for contrastive image and text pretraining.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains
    the encoders, while the loss is implemented in the class CLIPPretrainingLoss.


    Args:   vision_encoder (nn.Module): Instantiated vision encoder.
                See CLIPVisionEncoder class.
            text_encoder (nn.Module): Instantiated text encoder.
                See CLIPTextEncoder class.

    Inputs: image (Tensor): Tensor containing image features.
            text (Tensor): Tensor containing text features.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:

        img_embeddings = self.vision_encoder(image)
        text_embeddings = self.text_encoder(text)
        img_embeddings = F.normalize(img_embeddings)
        text_embeddings = F.normalize(text_embeddings)
        return {"image": img_embeddings, "text": text_embeddings}


def clip_vit_b16():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,  # based on https://git.io/JMpJK
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    return CLIPModel(vision_encoder, text_encoder)


def clip_vit_b32():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=512)
    return CLIPModel(vision_encoder, text_encoder)


def clip_vit_l14():
    vision_encoder = VisionTransformer(
        image_size=224,
        patch_size=14,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        num_classes=768,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=768, width=768, heads=12)
    return CLIPModel(vision_encoder, text_encoder)


def clip_rn50():
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 6, 3),
        output_dim=1024,
        heads=1024,
        width=2048,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    return CLIPModel(vision_encoder, text_encoder)


def clip_rn101():
    vision_encoder = ResNetForCLIP(
        layers=(3, 4, 23, 3),
        output_dim=1024,
        heads=1024,
        width=2048,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024)
    return CLIPModel(vision_encoder, text_encoder)


# Note: these models require larger image sizes
def clip_rn50x4():
    vision_encoder = ResNetForCLIP(
        layers=(4, 6, 10, 6),
        output_dim=640,
        heads=1280,
        input_resolution=288,
        width=2560,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024, width=640, heads=12)
    return CLIPModel(vision_encoder, text_encoder)


def clip_rn50x16():
    vision_encoder = ResNetForCLIP(
        layers=(6, 8, 18, 8),
        output_dim=768,
        heads=1536,
        input_resolution=384,
        width=3072,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=768, width=768, heads=12)
    return CLIPModel(vision_encoder, text_encoder)


def clip_rn50x64():
    vision_encoder = ResNetForCLIP(
        layers=(3, 15, 36, 10),
        output_dim=1024,
        heads=2048,
        input_resolution=448,
        width=4096,
    )
    text_encoder = CLIPTextEncoder(embedding_dim=1024, width=1024, heads=16)
    return CLIPModel(vision_encoder, text_encoder)


# Note: these models use torchvision's ResNet
def clip_rn50_tv():
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        num_classes=1024,
    )
    text_encoder = CLIPTextEncoder()
    return CLIPModel(vision_encoder, text_encoder)


def clip_rn101_tv():
    vision_encoder = ResNet(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        num_classes=512,
    )
    text_encoder = CLIPTextEncoder()
    return CLIPModel(vision_encoder, text_encoder)
