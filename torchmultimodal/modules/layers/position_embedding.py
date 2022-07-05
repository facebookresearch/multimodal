# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Tuple

import torch
from torch import nn, Tensor


class BroadcastedPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Based on broadcasted position embedding algorithm in codebase:
        https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458

        Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        shape (Tuple[int, ...]): shape of raw data before batching and embedding
        embedding_dim (int): the size of each embedding vector

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``

    Inputs:
        position_ids (Tensor): 1D tensor of integers indicating locations of the broadcasted
            position embeddings to be returned.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if embedding_dim % len(shape) != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} modulo len(shape) {len(shape)} is not zero"
            )

        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embedding_dim = embedding_dim

        self.embedding = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(
                    torch.randn(shape[i], embedding_dim // n_dim) * 0.01
                )
                for i in range(n_dim)
            }
        )

    @property
    def indices(self) -> Tensor:
        """Returns broadcasted indices of the data

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 3), embedding_dim=6)
            >>> pos_emb.indices
            tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        """
        return torch.cartesian_prod(*[torch.arange(s) for s in self.shape])

    def _broadcast(self, i: int) -> Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.shape``. The embedding dim is not touched.

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"].shape)
            torch.Size([2, 3])
            >>> pos_emb.embedding["d_0"] = nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
            >>> out = pos_emb._broadcast(i=0)
            >>> print(out)
            tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]])
            >>> print(out.shape)
            (2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f"d_{i}"]
        # (1, ..., 1, self.shape[i], 1, ..., embedding_dim)
        emb = emb.view(
            *itertools.repeat(1, i),
            self.shape[i],
            *itertools.repeat(1, (self.n_dim - i - 1)),
            -1,
        )
        # (*self.shape, embedding_dim)
        emb = emb.expand(*self.shape, -1)

        return emb

    def forward(self, position_ids: Tensor) -> Tensor:
        if torch.max(position_ids) >= len(self.indices) or torch.min(position_ids) < -1:
            raise IndexError(f"Invalid position ids: {position_ids}")

        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)

        # concatenated embeddings: (*(shape), embedding_dim)
        embeddings = torch.cat(embeddings, dim=-1)
        # extract relevant indices for the embeddings and transpose: (len(position_ids), len(shape))
        indices = [*self.indices[position_ids].transpose(0, 1)]
        # return the relevant embeddings: (len(position_ids), embedding_dim)
        # the embeddings are flattened across ``shape`` as the indices are broadcasted
        embeddings = embeddings[indices]

        # (batch, len(position_ids), embedding_dim)
        return embeddings.unsqueeze(0)
