# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchtext.models.roberta.modules import TransformerEncoder


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder class. Should be instantiated and passed to
    CLIPArchitecture (architectures/clip.py)

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

    def initialize_parameters(self) -> None:
        # Initialize token and positional embeddings
        nn.init.normal_(
            self.encoder.token_embedding.weight, std=self.TOKEN_EMBEDDING_INIT_STD
        )
        nn.init.normal_(
            self.encoder.positional_embedding.embedding.weight,
            std=self.POS_EMBEDDING_INIT_STD,
        )

        proj_std = (self.width**-0.5) * ((2 * self.encoder.layers.num_layers) ** -0.5)
        attn_std = self.width**-0.5
        fc_std = (2 * self.width) ** -0.5
        for layer in self.encoder.layers.layers:
            nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
            # c_fc in CLIP corresponds to the first residual MLP layer
            nn.init.normal_(layer.linear1.weight, std=fc_std)
            # c_proj in CLIP corresponds to the last residual MLP layer
            nn.init.normal_(layer.linear2.weight, std=proj_std)

        # Initialize projection
        nn.init.normal_(self.projection.weight, std=self.width**-0.5)

    def build_attention_mask(self) -> torch.Tensor:
        # To support torchscripting, we have to pass an int as fill_value
        mask = torch.full((self.context_length, self.context_length), int(True)).triu(1)
        return mask.to(device=None, dtype=torch.bool)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(text, attn_mask=self.build_attention_mask())
        # To support torchscripting, embeddings must explicitly be a Tensor and not List[Tensor]
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("`embeddings` must be of type Tensor.")
        # [n_ctx, bs, transformer.width] -> [bs, n_ctx, transformer.width]
        embeddings = torch.permute(embeddings, (1, 0, 2))
        embeddings = self.ln_final(embeddings)
        # take features from the eot embedding (the highest number in each sequence)
        embeddings = self.projection(
            embeddings[torch.arange(embeddings.shape[0]), text.argmax(dim=-1)]
        )
        # embeddings now has size [bs, embedding_dim]

        return embeddings
