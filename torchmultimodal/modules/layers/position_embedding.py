# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from typing import List, Tuple

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


class AlibiPositionEmbeddings(nn.Module):
    """Attention with Linear Biases (ALiBi)

    # Softmax(qiKT + m · [-(i - 1), ..., -2, -1, 0]),
    where m = fixed specific slope per head

    as proposed in:
    https://arxiv.org/abs/2108.12409
    Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation

    derived from Ofir Press (alibi author) codebase:
    https://github.com/ofirpress/attention_with_linear_biases

    """

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
    ) -> None:
        """recommended usage:  create alibi mask before transformer block loop and integrate
        Alibi should be applied after the sqrt scaling of the attention values

        Example:
        before Transformer block loop:
            from alibi_embeddings import AlibiPE
            self.alibi = AlibiPE(config.max_seq_len, config.num_heads)
        pass a reference to the alibi class to each transformer layer
        then in forward of transformer layer:
            alibi_mask = self.alibi.get_attention_mask(N) # N = seq length of this batch
            ...
            attn = q @ k.transpose( -2, -1)
            att *= 1.0 / math.sqrt(k.size(-1))
            att += alibi_mask

        """
        super().__init__()

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.causal_mask = self.build_causal_attention_mask(
            self.max_seq_len, self.num_heads
        )
        self.alibi_mask_base = self.build_alibi_mask(self.max_seq_len, self.num_heads)
        self.decoder_mask = self.causal_mask + self.alibi_mask_base
        self.register_buffer("alibi_mask", self.decoder_mask, persistent=False)

    def get_attention_mask(self, curr_seq_len: int) -> torch.Tensor:
        """returns the alibi mask, clipped to the current batch seq len"""
        return self.alibi_mask[..., :curr_seq_len, :curr_seq_len]

    @classmethod
    def build_causal_attention_mask(cls, seq_len: int, num_heads: int) -> torch.Tensor:
        """builds a generic causal attention mask"""
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )
        attn_mask = causal_mask.repeat(num_heads, 1, 1)
        return attn_mask

    @classmethod
    def build_alibi_mask(cls, seq_len: int, num_heads: int) -> torch.Tensor:
        """generate the alibi mask by computing a distance bias matrix multiplied by each head's m (slope)"""
        distance_bias_matrix = -torch.abs(
            torch.arange(seq_len) - torch.arange(seq_len).view(-1, 1)
        )
        slope_per_head = Tensor(cls.get_slopes(num_heads)).view(-1, 1, 1)
        alibi_mask = distance_bias_matrix * slope_per_head
        return alibi_mask

    @staticmethod
    def get_slopes(num_heads: int) -> List[float]:
        """for n heads, a range from (0,1) and is the geometric sequence
        that starts at 2^(-8/n) and uses this same value as its ratio

        example: num_heads =4
        result: [0.25, 0.0625, 0.015625, 0.00390625]

        """

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)

        # paper authors note that they only trained models that have 2^a heads for some a.
        # This has beneficial properties related to input being power of 2.

        # Closest power of 2 below is workaround for when num of heads is not power of 2
        # Slopes are returned in ordered sequence to keep symmetry.

        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

        a = get_slopes_power_of_2(closest_power_of_2)
        b = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
            : num_heads - closest_power_of_2
        ]
        return [x for pair in zip(b, a) for x in pair] + a[len(b) :]
