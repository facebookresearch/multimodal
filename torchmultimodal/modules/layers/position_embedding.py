# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchmultimodal.utils.common import tensor_slice


# Reference:
# https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458
class BroadcastedPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        shape (Tuple[int, ...]): shape of raw data before batching and embedding
        embedding_dim (int): the size of each embedding vector

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``

    Inputs:
        x (Optional[Tensor]): flattened input data, e.g., ``(batch, time * height * width, embedding_dim)``.
        decode_step (Optional[int]): position of the data that requires decoding.
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
    def decode_idxs(self) -> List:
        """Indices along the dims of data, e.g., ``(time, height, width)``."""
        return list(itertools.product(*[range(s) for s in self.shape]))

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
            tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]])
            >>> print(out.shape)
            (1, 2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f"d_{i}"]
        # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
        emb = emb.view(
            1,
            *itertools.repeat(1, i),
            self.shape[i],
            *itertools.repeat(1, (self.n_dim - i - 1)),
            -1,
        )
        # (1, *self.shape, -1)
        emb = emb.expand(1, *self.shape, -1)

        return emb

    def _decode(
        self, decode_step: int, embeddings: Tensor, x_shape: Tuple[int, ...]
    ) -> Tensor:
        """Returns the embedding vector immediately before the decoding location."""
        decode_idx = self.decode_idxs[decode_step - 1]
        embeddings = tensor_slice(
            embeddings,
            [0, *decode_idx, 0],
            [x_shape[0], *itertools.repeat(1, self.n_dim), x_shape[-1]],
        )

        return embeddings

    def forward(
        self, x: Optional[Tensor] = None, decode_step: Optional[int] = None
    ) -> Tensor:
        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)

        embeddings = torch.cat(
            embeddings, dim=-1
        )  # concatenated embeddings: (1, *(shape), embedding_dim)

        if decode_step is not None:
            embeddings = self._decode(decode_step, embeddings, tuple(x.shape))
            # decoded embedding: (1, *repeat(1, len(shape)), embedding_dim)

        return embeddings.flatten(start_dim=1, end_dim=-2)
