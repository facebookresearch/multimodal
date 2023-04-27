# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn, Tensor

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchmultimodal.modules.layers.activation import SiLU
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder class. Should be instantiated and passed to
    CLIP (models/clip.py)

    As in CLIP, the text encoder follows a Transformer architecture.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        context_length (int): Maximum sequence length for Transforer.
        vocab_size (int): Vocab size.
        width (int): Embedding dimension for Transformer encoder.
        dim_feedforward (int): Dimension of the feedfoward networks.
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
        dim_feedforward: int = 2048,
        heads: int = 8,
        layers: int = 12,
        use_clip_init: bool = True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.token_embedding = torch.nn.Embedding(vocab_size, width)
        self.positional_embedding = torch.nn.Parameter(
            torch.empty(context_length, width)
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=width,
            dim_feedforward=dim_feedforward,
            nhead=heads,
            dropout=0.0,
            activation=SiLU(),
            norm_first=True,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=layers)

        self.width = width
        self.context_length = context_length

        self.ln_final = Fp32LayerNorm(width)
        self.projection = nn.Linear(width, embedding_dim, bias=False)

        self.mask = torch.full(
            (self.context_length, self.context_length),
            float("-inf"),
        ).triu(1)

        if use_clip_init:
            self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Initialize token and positional embeddings
        nn.init.normal_(self.token_embedding.weight, std=self.TOKEN_EMBEDDING_INIT_STD)
        nn.init.normal_(
            self.positional_embedding,
            std=self.POS_EMBEDDING_INIT_STD,
        )

        proj_std = (self.width**-0.5) * ((2 * self.encoder.num_layers) ** -0.5)
        attn_std = self.width**-0.5

        fc_std = (2 * self.width) ** -0.5

        for layer in self.encoder.layers:
            nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
            # c_fc in CLIP corresponds to the first residual MLP layer
            nn.init.normal_(layer.linear1.weight, std=fc_std)
            # c_proj in CLIP corresponds to the last residual MLP layer
            nn.init.normal_(layer.linear2.weight, std=proj_std)

        # Initialize projection
        nn.init.normal_(self.projection.weight, std=self.width**-0.5)

    def build_attention_mask(self) -> Tensor:
        # To support torchscripting, we have to pass an int as fill_value
        mask = torch.full(
            (self.context_length, self.context_length), float("-inf")
        ).triu(1)
        return mask

    def forward(self, text: Tensor) -> Tensor:
        if text.size(1) != self.context_length:
            raise ValueError(
                f"length of input should be {self.context_length} but found {text.size(1)}"
            )
        embeddings = self.token_embedding(text)
        embeddings = embeddings + self.positional_embedding
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)

        # [n_ctx, bs, transformer.width] -> [bs, n_ctx, transformer.width]
        embeddings = torch.permute(embeddings, (1, 0, 2))
        embeddings = self.ln_final(embeddings)
        # take features from the eot embedding (the highest number in each sequence)
        embeddings = self.projection(
            embeddings[torch.arange(embeddings.shape[0]), text.argmax(dim=-1)]
        )
        # embeddings now has size [bs, embedding_dim]

        return embeddings
