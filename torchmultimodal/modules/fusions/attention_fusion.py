# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
from torch import nn, Tensor


class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """

    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        encoding_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(
            nn.Linear(attn_in_dim, len(channel_to_encoder_dim)),
            nn.Softmax(-1),
        )
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())

        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(
                channel_to_encoder_dim[channel], encoding_projection_dim
            )
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:
        concatenated_in = torch.cat(
            [embeddings[k] for k in sorted(embeddings.keys())], dim=-1
        )
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: List[Tensor] = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)

        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = (
                attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
            )

        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused
