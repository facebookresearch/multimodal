# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class DeepsetFusionModule(nn.Module):
    """
    Fuse embeddings through stacking followed by pooling strategy and MLP
    See https://arxiv.org/pdf/2003.01607.pdf

    Args:
        channel_to_encoder_dim (Dict[str, int]): mapping of channel name to the\
        encoding dimension
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).\
        Use MLP for mlp_classifier for default mlp.
        pooling_function (Callable): Pooling function to combine the tensors,\
        like torch.median\
        apply_attention (bool): If self attention (2 layer net) is applied before\
        stacking embeddings, defaults to False.
        attention_dim (int): intermediate dim for attention layer.\
        defaults to projection dim / 2
        modality_normalize (bool): If normalization is applied along the modality axis,\
        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim \
        is applied to the embeddings. defaults to False

    """

    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        mlp: nn.Module,
        pooling_function: Callable,
        apply_attention: bool = False,
        attention_dim: Optional[int] = None,
        modality_normalize: bool = False,
        norm_factor: float = 2.0,
        use_auto_mapping: bool = False,
    ):
        super().__init__()
        self.apply_attention = apply_attention
        self.modality_normalize = modality_normalize
        self.norm_factor = norm_factor
        self.use_auto_mapping = use_auto_mapping
        projection_dim = DeepsetFusionModule.get_projection_dim(
            channel_to_encoder_dim, use_auto_mapping
        )
        if self.use_auto_mapping:
            self.projections = nn.ModuleDict(
                {
                    channel: nn.Linear(dim, projection_dim)
                    for channel, dim in channel_to_encoder_dim.items()
                }
            )
        else:
            self.projections = nn.ModuleDict(
                {channel: nn.Identity() for channel in channel_to_encoder_dim}
            )
        if self.apply_attention:
            self.attention: nn.Module
            if attention_dim is None:
                # default value as per older implementation
                attention_dim = projection_dim // 2
            self.attention = nn.Sequential(
                nn.Linear(projection_dim, attention_dim),
                nn.Tanh(),
                nn.Linear(attention_dim, 1),
                # channel axis
                nn.Softmax(dim=-2),
            )
        else:
            self.attention = nn.Identity()

        self.pooling_function = pooling_function
        self.mlp = mlp

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:

        projections = {}
        for channel, projection in self.projections.items():
            projections[channel] = projection(embeddings[channel])

        embedding_list = [projections[k] for k in sorted(projections.keys())]

        # bsz x channels x projected_dim
        stacked_embeddings = torch.stack(embedding_list, dim=1)

        if self.apply_attention:
            attn_weights = self.attention(stacked_embeddings)
            stacked_embeddings = stacked_embeddings * attn_weights

        if self.modality_normalize:
            normalized_embeddings = F.normalize(
                stacked_embeddings, p=self.norm_factor, dim=1
            )
        else:
            normalized_embeddings = F.normalize(
                stacked_embeddings, p=self.norm_factor, dim=2
            )

        pooled_features = self._pool_features(normalized_embeddings)
        fused = self.mlp(pooled_features)
        return fused

    @classmethod
    def get_projection_dim(
        cls, channel_to_encoder_dim: Dict[str, int], use_auto_mapping: bool
    ) -> int:
        if use_auto_mapping:
            projection_dim = min(channel_to_encoder_dim.values())
        else:
            encoder_dim = set(channel_to_encoder_dim.values())
            if len(encoder_dim) != 1:
                raise ValueError(
                    "Encoder dimension should be same for all channels \
                    if use_auto_mapping is set to false"
                )
            projection_dim = encoder_dim.pop()
        return projection_dim

    def _pool_features(self, embeddings: Tensor) -> Tensor:
        pooled_embeddings = self.pooling_function(embeddings, dim=1)
        if torch.jit.isinstance(pooled_embeddings, Tuple[Tensor, Tensor]):
            return pooled_embeddings.values
        if not isinstance(pooled_embeddings, Tensor):
            raise ValueError(
                f"Result from pooling function should be a tensor.\
             {self.pooling_function} does not satisfy that"
            )
        return pooled_embeddings


class DeepsetFusionWithTransformer(DeepsetFusionModule):
    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        mlp: nn.Module,
        pooling_function: nn.TransformerEncoder,
        apply_attention: bool = False,
        attention_dim: Optional[int] = None,
        modality_normalize: bool = False,
        norm_factor: float = 2.0,
        use_auto_mapping: bool = False,
    ):
        super().__init__(
            channel_to_encoder_dim,
            mlp,
            pooling_function,
            apply_attention,
            attention_dim,
            modality_normalize,
            norm_factor,
            use_auto_mapping,
        )

    def _pool_features(self, embeddings: Tensor) -> Tensor:
        pooled = self.pooling_function(embeddings)
        # take representation of the first token as the pooled feature
        return pooled[:, 0, :]


def deepset_transformer(
    channel_to_encoder_dim: Dict[str, int],
    mlp: nn.Module,
    apply_attention: bool = False,
    attention_dim: Optional[int] = None,
    modality_normalize: bool = False,
    norm_factor: float = 2.0,
    use_auto_mapping: bool = False,
    num_transformer_att_heads: int = 8,
    num_transformer_layers: int = 1,
) -> nn.Module:
    """
    Helper wrapper function around DeepsetFusionWithTransformer, \
    to instantiate the transformer and pass it to the fusion module
    Args:
        channel_to_encoder_dim (Dict[str, int]): mapping of channel name to the\
        encoding dimension
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).\
        Use MLP for mlp_classifier for default mlp.
        pooling_function (Callable): Pooling function to combine the tensors,\
        like torch.median
        apply_attention (bool): If self attention is applied before\
        stacking embeddings, defaults to False
        attention_dim (int): intermediate dim for attention layer. \
        defaults to projection dim / 2
        modality_normalize (bool): If normalization is applied along the modality axis,\
        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim \
        is applied to the embeddings. defaults to False
        num_transformer_att_heads (int): number of attention heads. \
        Used only if pooling function set to transformer
        num_transformer_layers (int): number of transformer layers,\
        used only if pooling function set to transformer

    """
    projection_dim = DeepsetFusionWithTransformer.get_projection_dim(
        channel_to_encoder_dim, use_auto_mapping
    )
    if projection_dim % num_transformer_att_heads != 0:
        raise ValueError(
            f"projection dim should be divisible by attention heads\
                found {projection_dim} and {num_transformer_att_heads}"
        )
    transformer = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=projection_dim, nhead=num_transformer_att_heads, batch_first=True
        ),
        num_layers=num_transformer_layers,
        norm=nn.LayerNorm(projection_dim),
    )
    fusion = DeepsetFusionWithTransformer(
        channel_to_encoder_dim,
        mlp,
        transformer,
        apply_attention,
        attention_dim,
        modality_normalize,
        norm_factor,
        use_auto_mapping,
    )
    return fusion
