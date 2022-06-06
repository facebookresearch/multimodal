# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Optional

import numpy as np

import torch
import torch.nn as nn

from torchmultimodal.utils.common import tensor_slice


class BroadcastPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        shape (torch.Size): shape of raw data before batching and embedding
        embedding_dim (int): the size of each embedding vector
        dim (Optional[int]): dimension of embedding w.r.t. the other dimensions of batched data, e.g., ``-1``
            corresponds to embedding being the last dimension. Default is ``-1``.

    Raises:
        ValueError: if ``dim`` is neither ``-1`` nor ``1``
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``

    Inputs:
        x (Optional[torch.Tensor]): flattened input data, e.g., ``(batch, time * height * width, embedding_dim)``
            if ``dim=-1``.
        decode_step (Optional[int]): position of the data that requires decoding.
    """

    def __init__(
        self, shape: torch.Size, embedding_dim: int, dim: Optional[int] = -1
    ) -> None:
        super().__init__()
        if dim not in [-1, 1]:
            raise ValueError(
                f"Only first or last for the embedding dim is supported but got {dim}"
            )
        if embedding_dim % len(shape) != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} modulo len(shape) {len(shape)} is not zero"
            )

        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embedding_dim = embedding_dim
        self.dim = dim

        self.embedding = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(
                    torch.randn(shape[i], embedding_dim // n_dim) * 0.01
                    if dim == -1
                    else torch.randn(embedding_dim // n_dim, shape[i]) * 0.01
                )
                for i in range(n_dim)
            }
        )

    @property
    def seq_len(self) -> int:
        """Dimension of flattened data, e.g., time * height * width"""
        return np.prod(self.shape)

    @property
    def decode_idxs(self):
        """Indices along the dims of data, e.g., ``(time, height, width)``."""
        return list(itertools.product(*[range(s) for s in self.shape]))

    def _broadcast(self, emb: torch.Tensor, i: int) -> torch.Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.shape``. The embedding dim is kept at ``self.dim``.

        For example::

            >>> pos_emb = BroadcastPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"]).shape
            torch.Size([2, 3])
            >>> out = pos_emb._broadcast(torch.tensor([[0, 0, 0],[0, 0, 1]]), i=0)
            >>> print(out)
            tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]])

        The input is broadcasted along dim ``4`` since it's the ``0``-th embedding constructed w.r.t dim ``2``.
        """
        if self.dim == -1:
            # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
            emb = emb.view(
                1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1
            )
            # (1, *self.shape, -1)
            emb = emb.expand(1, *self.shape, -1)
        else:
            emb = emb.view(
                1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1))
            )
            emb = emb.expand(1, -1, *self.shape)

        return emb

    def _decode(
        self, decode_step: int, embeddings: torch.Tensor, x_shape: torch.Size
    ) -> torch.Tensor:
        """Returns the embedding vector immediately before the decoding location."""
        decode_idx = self.decode_idxs[decode_step - 1]
        if self.dim == -1:
            embeddings = tensor_slice(
                embeddings,
                [0, *decode_idx, 0],
                [x_shape[0], *(1,) * self.n_dim, x_shape[-1]],
            )
        elif self.dim == 1:
            embeddings = tensor_slice(
                embeddings,
                [0, 0, *decode_idx],
                [x_shape[0], x_shape[1], *(1,) * self.n_dim],
            )

        return embeddings

    def forward(
        self, x: Optional[torch.Tensor] = None, decode_step: Optional[int] = None
    ) -> torch.Tensor:
        embeddings = []
        for i in range(self.n_dim):
            emb = self.embedding[f"d_{i}"]
            emb = self._broadcast(emb, i)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=self.dim)

        if decode_step is not None:
            embeddings = self._decode(decode_step, embeddings, x.shape)

        return embeddings.flatten(start_dim=1, end_dim=-2)
