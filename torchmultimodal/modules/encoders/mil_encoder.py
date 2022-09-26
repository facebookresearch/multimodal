# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Union

import torch
from torch import nn, Tensor
from torchmultimodal.modules.fusions.deepset_fusion import (
    DeepsetFusionModule,
    DeepsetFusionWithTransformer,
)


class MILEncoder(nn.Module):
    """
    Multi instance learning encoder that partitions the input into a set of inputs
    and uses a shared encoder followed by deepset
    fusion to get a pooled representation of the entire input. Example use is to build a
    single representation from embeddings of all images in a post.

    Args:
        partition_sizes (List[int]): list of size for each partition of the input
        shared_encoder (nn.Module): Shared encoder for each partition of the input.
        shared_encoder_dim (int) : Output dimension of the encoders
        Following fields are same as the params for deepset fusion
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).\
        Use MLP from mlp_classifier for default mlp implementation.
        pooling_function (Callable): Pooling function to combine the tensors,\
        like torch.median
        apply_attention (bool): If self attention is applied before\
        stacking embeddings, defaults to False
        modality_normalize (bool): If normalization is applied along the modality axis,\
        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim \
        is applied to the embeddings. defaults to False

    """

    def __init__(
        self,
        partition_sizes: List[int],
        shared_encoder: nn.Module,
        shared_encoder_dim: int,
        mlp: nn.Module,
        pooling_function: Callable,
        apply_attention: bool = False,
        attention_dim: Optional[int] = None,
        modality_normalize: bool = False,
        norm_factor: float = 2.0,
        use_auto_mapping: bool = False,
    ):
        super().__init__()
        self.partition_sizes = partition_sizes
        self.shared_encoder = shared_encoder
        channel_to_encoder_dim = {}
        for i in range(len(partition_sizes)):
            channel_to_encoder_dim[self.get_channel_name(i)] = shared_encoder_dim
        deepset_fusion_cls = (
            DeepsetFusionWithTransformer
            if isinstance(pooling_function, nn.TransformerEncoder)
            else DeepsetFusionModule
        )

        self.deepset_fusion: Union[
            DeepsetFusionWithTransformer, DeepsetFusionModule
        ] = deepset_fusion_cls(
            channel_to_encoder_dim=channel_to_encoder_dim,
            mlp=mlp,
            pooling_function=pooling_function,  # type: ignore
            apply_attention=apply_attention,
            attention_dim=attention_dim,
            modality_normalize=modality_normalize,
            norm_factor=norm_factor,
            use_auto_mapping=use_auto_mapping,
        )

    def get_channel_name(self, id: int) -> str:
        # create dummy channel name to pass to fusion
        return f"mil_{id}"

    def forward(self, x: Tensor) -> Tensor:
        idx = 0
        input_size = x.size(dim=1)
        if input_size != sum(self.partition_sizes):
            raise ValueError(
                f"partition sizes should sum to the input size {input_size}"
            )
        partitioned_input = torch.split(x, self.partition_sizes, dim=1)

        encoded_input = {}
        for idx, input in enumerate(partitioned_input):
            key = self.get_channel_name(idx)
            encoded_input[key] = self.shared_encoder(input)

        return self.deepset_fusion(encoded_input)
