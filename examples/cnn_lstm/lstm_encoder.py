# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """An LSTM encoder. Stacks an LSTM on an embedding layer.

    Args:
        vocab_size (int): The size of the vocab for embeddings.
        embedding_dim (int): The size of each embedding vector.
        input_size (int): The number of features in the LSTM input.
        hidden_size (int): The number of features in the hidden state.
        bidirectional (bool): Whether to use bidirectional LSTM.
        batch_first (bool): Whether to provide batches as (batch, seq, feature)
            or (seq, batch, feature).

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        input_size: int,
        hidden_size: int,
        bidirectional: bool,
        batch_first: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.lstm(self.embedding(x))
        # N x B x H => B x X x H where N = num_layers * num_directions
        x = x[0].transpose(0, 1)

        # N should be 2 so we can merge in that dimension
        assert x.size(1) == 2, "hidden state (final) should have 1st dim as 2"

        x = torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1)
        return x
