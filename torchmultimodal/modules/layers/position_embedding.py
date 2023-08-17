# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Tuple, Union

import torch
from torch import nn, Tensor


class BroadcastedPositionEmbedding(nn.Module):
    r"""Spatiotemporal broadcasted positional embeddings.

    Based on broadcasted position embedding algorithm in codebase:
        https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458

        Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \prod_{j>i}\text{dim}[j]`.

    Args:
        latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
        embedding_dim (int): The size of each embedding vector.

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        """
        Args:
            latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
            embedding_dim (int): The size of each embedding vector.

        Raises:
            ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``
        """
        super().__init__()
        if embedding_dim % len(latent_shape) != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} modulo len(latent_shape) {len(latent_shape)} is not zero"
            )

        self.latent_shape = latent_shape
        self.n_dim = n_dim = len(self.latent_shape)
        self.embedding_dim = embedding_dim

        self.embedding = nn.ParameterDict(
            {
                f"d_{i}": nn.Parameter(
                    torch.randn(self.latent_shape[i], embedding_dim // n_dim) * 0.01
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
        return torch.cartesian_prod(*[torch.arange(s) for s in self.latent_shape])

    def _broadcast(self, i: int) -> Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.latent_shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.latent_shape``. The embedding dim is not touched.

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
        # (1, ..., 1, self.latent_shape[i], 1, ..., embedding_dim)
        emb = emb.view(
            *itertools.repeat(1, i),
            self.latent_shape[i],
            *itertools.repeat(1, (self.n_dim - i - 1)),
            -1,
        )
        # (*self.latent_shape, embedding_dim)
        emb = emb.expand(*self.latent_shape, -1)

        return emb

    def forward(self, position_ids: Tensor) -> Tensor:
        """
        Args:
            position_ids (Tensor): batches of of 1D integer tensors indicating locations of the broadcasted
                position embeddings to be returned.

        Returns:
            A tensor with the position embeddings selected by position ids.

        Raises:
            IndexError: If any position id(s) provided is outside of the indices range.
        """
        invalid_ids = position_ids[
            torch.logical_or(position_ids >= len(self.indices), position_ids < -1)
        ]
        if len(invalid_ids):
            raise IndexError(f"Invalid position ids: {invalid_ids}")

        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)

        # concatenated embeddings: (*(shape), embedding_dim)
        embeddings = torch.cat(embeddings, dim=-1)

        # expand the permuted tensor to form a list of size `n_dim`
        # where each elm is a tensor of shape (pos_ids, batch)
        indices = [*self.indices[position_ids].permute(2, 1, 0)]
        embeddings = embeddings[indices].transpose(0, 1)  # (batch, pos_ids, emb_dim)

        return embeddings


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for use of encoding timestep
    in diffusion models. Timestep is mapped onto positions on different
    frequencies of sinusoidal waveforms. Slightly different than the original
    Transformer implementation in paper "Attention Is All You Need".
    Taken from code of original author of DDPM paper, Ho et al. 2020.

    Code ref: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    Attributes:
        embed_dim (int): dimensionality of position embeddings. Default is 128, from original DDPM.

    Args:
        t (Tensor): Tensor of input timesteps of shape (batch, ).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        # Half of embedding dim is sin, half is cos
        half_dim = self.embed_dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.embed_dim % 2 == 1:
            embeddings = nn.functional.pad(embeddings, (0, 1))
        return embeddings


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        ratio: int = 10000,
        device=None,
    ):
        """
        Implements Rotary Positional Embeddings (RoPE)
        proposed in: https://arxiv.org/abs/2104.09864

        Args
        ----
        dim : int
            Per-head embedding dimension
        max_position_embeddings : int
            Maximum expected sequence length for the model, if exceeded the cached freqs will be recomputed
        ratio: int
            The ratio for the geometric progression to compute the rotation angles
        """
        super().__init__()
        self.register_buffer(
            "freqs",
            1.0
            / (
                ratio
                ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
            ),
        )
        self.compute_freqs_cis(max_position_embeddings)

    def compute_freqs_cis(self, max_position_embeddings=2048):
        t = torch.arange(
            max_position_embeddings, device=self.freqs.device, dtype=self.freqs.dtype
        )
        freqs = torch.outer(t, self.freqs).float()
        self.max_seq_len_cached = max_position_embeddings
        self.register_buffer(
            "cached_freqs",
            torch.stack(
                [
                    torch.cos(freqs),
                    -torch.sin(freqs),
                    torch.sin(freqs),
                    torch.cos(freqs),
                ],
                dim=2,
            ).view(*freqs.shape, 2, 2),
        )

    def reshape_for_broadcast(
        self, x: torch.Tensor, cur_freqs: torch.Tensor
    ) -> torch.Tensor:
        ndim = x.ndim
        assert 1 < ndim
        assert cur_freqs.shape[:2] == (x.shape[2], x.shape[-2])
        shape = [d if i == 2 or i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
        return cur_freqs.view(*shape, 2)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, start_pos: Union[int, torch.LongTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        q : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        k : torch.Tensor
            Embedded query tensor, expected size is B x H x S x Eh
        start_pos : Union[int, torch.LongTensor]
            The starting position of the tokens encoded in q and k. This is important in
            kv-caching and left-padding situations, for which the rotation to be applied might
            not always be the pre-cached position 0...S. For kv-caching without dynamic batching
            start_pos is shared for all the batch.
        """
        seq_len = q.shape[2]
        q_ = q.float().reshape(*q.shape[:-1], -1, 2)  # B H L D/2 2
        k_ = k.float().reshape(*k.shape[:-1], -1, 2)  # B H L D/2 2

        if isinstance(start_pos, int):
            if start_pos + seq_len > self.max_seq_len_cached:
                self.compute_freqs_cis(start_pos + seq_len)
            cur_freqs = self.cached_freqs[start_pos : start_pos + seq_len]
            freqs = self.reshape_for_broadcast(q_, cur_freqs)
        else:
            max_start_pos = torch.max(start_pos).item()
            if max_start_pos + seq_len > self.max_seq_len_cached:
                self.compute_freqs_cis(max_start_pos + seq_len)
            freqs_idxs = torch.arange(0, seq_len, dtype=torch.long).repeat(
                start_pos.shape[0]
            ).view(-1, seq_len) + start_pos.view(-1, 1)
            freqs = self.cached_freqs[freqs_idxs].unsqueeze(1)

        freqs = freqs.float()  # 1 1 L D/2 2 2
        q_out = freqs.mul(q_.unsqueeze(-2)).sum(5).flatten(3)
        k_out = freqs.mul(k_.unsqueeze(-2)).sum(5).flatten(3)
        return q_out.type_as(q).contiguous(), k_out.type_as(k).contiguous()
