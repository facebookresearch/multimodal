# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchmultimodal.modules.layers.transformer_decoder import TransformerDecoder


class RightShift(nn.Module):
    """Shift the input sequence by 1 unit to the right and prepend with start of sentence token.

    Since the decoder progresses by taking the token it generates in the previous step, before it
    has generated anything it needs a token to start with. Hence, the start-of-sentence (SOS) token.
    The SOS token is a learnable parameter of the decoder and the choice of its initialization is taken
    from VideoGPT: https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py#L517
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(
            torch.FloatTensor(embedding_dim).normal_(std=0.02), requires_grad=True
        )

    def forward(self, x):
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


class MultimodalGPT(nn.Module):
    """GPT model for cross-modality generation"""

    def __init__(
        self,
        num_attention_heads: int,
        num_hidden_layers: int,
        dropout: float,
        attention_dropout: float,
        input_seq_len: int,
        output_seq_len: int,
        token_embedding_input: nn.Module,
        token_embedding_output: nn.Module,
        position_embedding_input: nn.Module,
        position_embedding_output: nn.Module,
    ) -> None:
        super().__init__()

        self._assert_modules()

        self.embedding_dim = token_embedding_input.embedding_dim
        self.right_shift = RightShift(self.embedding_dim)
        self.position_embedding_input = position_embedding_input
        self.position_embedding_output = position_embedding_output
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        total_len = self.input_seq_len + self.output_seq_len

        TransformerDecoder(
            hidden_size=embedding_dim,
            # The choice of ``intermediate_size`` follows OpenAI GPT2
            # Refernece: https://github.com/openai/gpt-2/blob/master/src/model.py#L128
            intermediate_size=[embedding_dim * 4],
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    # TODO: clean up this method
    def _assert_modules(self):
        if not hasattr(token_embedding_input, "embedding_dim"):
            raise AttributeError(
                "Token embedding of the input modality does not have attribute: embedding_dim!"
            )
        if not hasattr(token_embedding_output, "embedding_dim"):
            raise AttributeError(
                "Token embedding of the output modality does not have attribute: embedding_dim!"
            )
        if not hasattr(position_embedding_input, "embedding_dim"):
            raise AttributeError(
                "Position embedding of the input modality does not have attribute: embedding_dim!"
            )
        if not hasattr(position_embedding_output, "embedding_dim"):
            raise AttributeError(
                "Position embedding of the output modality does not have attribute: embedding_dim!"
            )
        if (
            token_embedding_input.embedding_dim
            != position_embedding_input.embedding_dim
        ):
            raise ValueError(
                "Token embedding dim must be equal to position embedding dim:"
                f"\t{token_embedding_input.embedding_dim} != {position_embedding_input.embedding_dim}"
            )
        if (
            position_embedding_input.embedding_dim
            != position_embedding_output.embedding_dim
        ):
            raise ValueError(
                f"Position embedding dim of the input modality {position_embedding_input.embedding_dim}"
                f"must be equal to that of the output modality {position_embedding_output.embedding_dim}"
            )

    # TODO: inference methods are candidates for generation mixin class
    # TODO: consider lowering them to TransformerDecoder after mixining
    def _get_inference_embeddings(self, x, position_embedding, decode_step):
        # Use the data point at ``decode_step - 1`` to generate the next.
        return x + position_embedding(x, decode_step)

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
        # Prepend start-of-sentence token to decode from
        if decode_step is not None and decode_step == 0:
            x = self.right_shift(x, decode_step)

        for net in self.attention_nets:
            x = net(x, decode_step, decode_idx)

        return x
