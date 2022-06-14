# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchmultimodal.modules.layers.attention import FullAttention, MultiHeadAttention
from torchmultimodal.modules.layers.mlp import MLP


class SiLU(nn.Module):
    r"""Sigmoind Linear Unit

    .. math:: \text{SiLU}(x) = x * \sigma(1.702 * x)

    where :math:`\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU implemented by pytorch for greater forward speed.
    Reference: `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class RightShift(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(
            torch.FloatTensor(embedding_dim).normal_(std=0.02), requires_grad=True
        )

    def forward(self, x, decode_step=None):
        if decode_step is not None and decode_step > 0:
            return x

        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2)  # (batch, seq_len, embedding_dim)
        sos = (
            torch.ones(x_shape[0], 1, self.embedding_dim, dtype=torch.float32).to(
                self.sos
            )
            * self.sos
        )  # (batch, 1, embedding_dim)
        sos = sos.type_as(x)
        # Shift one unit to the right along dim ``seq_len``
        x = torch.cat([sos, x[:, :-1, :]], axis=1)  # (batch, seq_len, embedding_dim)
        x = x.view(*x_shape)
        return x


class MLPBlock(MLP):
    """Transformer MLP block."""

    def __init__(self, in_dim, out_dim, hidden_dims):
        super().__init__(
            in_dim, out_dim, hidden_dims, 0.0, activation=SiLU, normalization=None
        )


class TransformerLayer(nn.Module):
    def __init__(
        self,
        shape,
        embedding_dim,
        num_attention_heads,
        num_hidden_layers,
        dropout,
        attention_dropout,
    ):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(embedding_dim)
        self.post_attention_dropout = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(
            shape,
            embedding_dim,
            embedding_dim,
            num_attention_heads,
            num_hidden_layers,
            causal=True,
            attention_module=FullAttention(shape, True, attention_dropout),
        )
        self.pre_mlp_norm = nn.LayerNorm(embedding_dim)
        self.post_mlp_dropout = nn.Dropout(dropout)
        self.mlp_block = MLPBlock(embedding_dim, embedding_dim, [embedding_dim * 4])

    def forward(self, x, decode_step, decode_idx):
        h = self.pre_attention_norm(x)
        if self.training:
            h = checkpoint(self.attention, h, h, h, decode_step, decode_idx)
        else:
            h = self.attention(h, h, h, decode_step, decode_idx)
        h = self.post_attention_dropout(h)
        x = x + h

        h = self.pre_mlp_norm(x)
        if self.training:
            h = checkpoint(self.mlp_block, h)
        else:
            h = self.mlp_block(h)
        h = self.post_mlp_dropout(h)
        x = x + h

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        position_embedding_input,
        position_embedding_output,
        num_attention_heads,
        num_hidden_layers,
        dropout,
        attention_dropout,
    ):
        super().__init__()
        if (
            position_embedding_input.embedding_dim
            != position_embedding_output.embedding_dim
        ):
            raise ValueError(
                f"The position embedding dimension of the input modality {position_embedding_input.embedding_dim}"
                f"must be equal to that of the output modality {position_embedding_output.embedding_dim}"
            )

        self.embedding_dim = position_embedding_input.embedding_dim
        self.right_shift = RightShift(self.embedding_dim)
        self.position_embedding_input = position_embedding_input
        self.position_embedding_output = position_embedding_output
        self.input_seq_len = self.position_embedding_input.seq_len
        self.output_seq_len = self.position_embedding_output.seq_len
        total_len = self.input_seq_len + self.output_seq_len

        self.attention_nets = nn.ModuleList(
            [
                TransformerLayer(
                    shape=(total_len),
                    embedding_dim=self.embedding_dim,
                    num_attention_heads=num_attention_heads,
                    num_hidden_layers=num_hidden_layers,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(num_hidden_layers)
            ]
        )

    def _get_inference_embeddings(self, x, position_embedding, decode_step):
        # Use the data point at ``decode_step - 1`` to generate the next.
        return position_embedding(x, decode_step)

    def _get_training_embeddings(self, x, position_embedding):
        # Use the full sequence for training
        return x + position_embedding(x)

    def forward(self, x_input, x_output, decode_step=None, decode_idx=None):
        # Trigger inference mode if either of the input/output-modality sequence is ``None`` as generation
        # is based the previous data point (autoregressive) where only one modality is needed at any point
        # along the sequence.
        if x_output is None:
            # Skip inferencing if ``decode_step = 0`` as decoding requires previous data point.
            # ``decode_step`` ranges between ``(0, input_seq_len]`` controlled by the GPT model.
            # When ``decode_step = input_seq_len``, the first output modality is generated from the last input
            # modality.
            if decode_step > 0:
                x_input = self._get_inference_embeddings(
                    x_input, self.position_embedding_input, decode_step
                )
            x = x_input
        elif x_input is None:
            # Continue to generate output-modality sequence from ``input_seq_len + 1``(inclusive) to
            # ``input_seq_len + output_seq_len - 1``(inclusive).
            x = self._get_inference_embeddings(
                x_output,
                self.position_embedding_output,
                decode_step - self.input_seq_len,
            )
        # Trigger training mode if both input/output-modality sequences are present as we know the ground
        # truth at each point.
        else:
            x_input = self._get_training_embeddings(
                x_input, self.position_embedding_input
            )
            x_output = self._get_training_embeddings(
                x_output, self.position_embedding_output
            )
            x = torch.cat((x_input, x_output), 1)
        x = self.right_shift(x, decode_step)

        for net in self.attention_nets:
            x = net(x, decode_step, decode_idx)

        return x
