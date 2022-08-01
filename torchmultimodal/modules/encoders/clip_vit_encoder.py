# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor

# Taken from original clip implementation https://github.com/openai/CLIP/blob/main/clip/model.py#L167
# TODO: unify with the implementation in text encoder
def quick_gelu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(1.702 * x)


class CLIPViTEncoder(nn.Module):
    """
    Vision transformer encoder for CLIP.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        patch_size (int): The dimension of each patch
        image_size(int): The size (width==height) of input image
        width (int): Dimensionality of the encoder layers and the pooler layer
        heads (int): Number of attention heads for each attention layer in the Transformer encoder
        layers (int): Number of hidden layers in the Transformer encoder

    Inputs:
        x (Tensor): image tensor with dimensions B x C(3) x image_size x image_size
    """

    def __init__(
        self,
        embedding_dim: int,
        patch_size: int,
        image_size: int,
        width: int,
        heads: int,
        layers: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.image_size = image_size

        scale = width**-0.5
        self.cls_token_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dropout=0.0,
            activation=quick_gelu,
            norm_first=True,
            dim_feedforward=4 * width,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:

        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(
                f"Expected input with width and height as {self.image_size}, found {x.size(2)} by {x.size(3)} "
            )
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels found {x.size(1)}")

        # B x C x image_size x image_size => B x C x patch_size x patch_size
        x = self.conv(x)

        # B x C x patch_size x patch_size => B x C x patch_size ** 2
        x = torch.flatten(x, start_dim=2)

        # B x C x patch_size ** 2 => B x patch_size ** 2 x C
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = self.encoder(x)

        x = self.ln_post(x[:, 0, :])
        x = x @ self.projection
        return x
