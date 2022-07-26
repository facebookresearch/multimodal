# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.attention import FullAttention, MultiHeadAttention
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.utils.attention import get_extended_attention_mask
from torchmultimodal.utils.common import checkpoint_wrapper


# TODO: Add docstring

# TODO: add right shift

# TODO: Implement generation utility in common util with RightShift
# TODO: Add text_video_gpt.py

"""
Old MUGEN implementation
class P2QAttentionStack:

    def __init__(self, embedding_dim):
        super().__init__()

        self.right_shift = RightShift(embedding_dim)

def forward(self, in_modality, out_modality, decode_step):
    if out_modality is None:
        # Skip inferencing if ``decode_step = 0`` as decoding requires previous data point.
        # ``decode_step`` ranges between ``(0, in_seq_len]`` controlled by the GPT model.
        # When ``decode_step = in_seq_len``, the first output modality is generated from the last input
        # modality.
        if decode_step > 0:
            in_modality = self._get_inference_embeddings(
                in_modality, self.in_pos_emb, decode_step
            )
        x = in_modality
    elif in_modality is None:
        # Continue to generate output-modality sequence from ``in_seq_len + 1``(inclusive) to
        # ``in_seq_len + out_seq_len - 1``(inclusive).
        x = self._get_inference_embeddings(
            out_modality,
            self.out_pos_emb,
            decode_step - self.in_seq_len,
        )
    # Trigger training mode if both input/output-modality sequences are present as we know the ground
    # truth at each point.
    else:
        in_modality = self._get_training_embeddings(
            in_modality, self.in_pos_emb
        )
        out_modality = self._get_training_embeddings(
            out_modality, self.out_pos_emb
        )
        x = torch.cat((in_modality, out_modality), 1)
    # Prepend start-of-sentence token to decode from
    if decode_step is not None and decode_step == 0:
        x = self.right_shift(x, decode_step)

    for net in self.attention_nets:
        x = net(x, decode_step, decode_idx)

    return x
"""


class TransformerDecoderOutput(NamedTuple):
    last_hidden_states: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attention_weights: Optional[Tuple[Tensor, ...]] = None
    past_key_values: Optional[Tuple[Dict[str, Tensor], ...]] = None


class TransformerLayerOutput(NamedTuple):
    hidden_states: Tensor
    attention_weights: Optional[Tensor] = None
    past_key_values: Optional[Dict[str, Tensor]] = None


class MultimodalGPT(nn.Module):
    """GPT model for cross-modality generation"""

    def __init__(
        self,
        in_token_emb: nn.Module,
        out_token_emb: nn.Module,
        in_pos_emb: nn.Module,
        out_pos_emb: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()

        self.in_token_emb = in_token_emb
        self.out_token_emb = out_token_emb
        self.in_pos_emb = in_pos_emb
        self.out_pos_emb = out_pos_emb
        self.decoder = decoder

    def forward(
        self,
        in_modality: Optional[Tensor] = None,
        out_modality: Optional[Tensor] = None,
        in_pos_ids: Optional[Tensor] = None,
        out_pos_ids: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
        device: Optional[str] = None,
    ) -> TransformerDecoderOutput:
        if (in_modality is None) and (out_modality is None):
            raise ValueError(
                "in_modality and out_modality sequences cannot be both empty"
            )

        # Since generation is based the previous data point (autoregressive) where
        # only one modality is needed at any point along the sequence, either input
        # or output modality can be None.
        # Whereas training is done by paralleling all data points so both modalities
        # should be present. Position ids are optional as they can be derived from
        # the sequence length of each modality.
        if in_modality is None:
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids, device)
            x = self._encode(out_modality, out_pos_ids, "out")
        elif out_modality is None:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids, device)
            x = self._encode(in_modality, in_pos_ids, "in")
        else:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids, device)
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids, device)
            x_in = self._encode(in_modality, in_pos_ids, "in")
            x_out = self._encode(out_modality, out_pos_ids, "out")
            x = torch.cat((x_in, x_out), dim=1)

        out = self.decoder(
            x,
            attn_mask,
            head_mask,
            use_cache,
            causal,
            return_attn_weights,
            return_hidden_states,
        )

        return TransformerDecoderOutput(
            out.last_hidden_states,
            out.hidden_states,
            out.attention_weights,
            out.past_key_values,
        )

    def _encode(self, x: Tensor, pos_ids: Tensor, modality: str) -> Tensor:
        token_emb = getattr(self, f"{modality}_token_emb")
        pos_emb = getattr(self, f"{modality}_pos_emb")
        _, x = token_emb(x)  # (tokens, embedding per token)
        x = x + pos_emb(pos_ids)

        return x

    def _norm_pos_ids(
        self, x: Tensor, pos_ids: Optional[Tensor] = None, device: Optional[str] = None
    ) -> Tensor:
        bs, seq_len, _ = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)[
                None, :
            ]  # (bs, seq_len)

        if pos_ids.shape != (bs, seq_len):
            raise ValueError(
                "input sequence and position ids must be equal in batch size and length"
            )

        return pos_ids


class TransformerDecoder(nn.Module):
    """A transformer decoder

    Attributes:
        decoder_layers (nn.Module):
        num_layers (int):

    Args:

    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerDecoderOutput:
        all_hidden_states = () if return_hidden_states else None
        all_attentions = () if return_attn_weights else None
        all_past_key_values = () if use_cache else None

        for layer in self.layers:
            if return_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attn_mask,
                head_mask,
                use_cache,
                causal,
                return_attn_weights,
            )
            hidden_states = layer_outputs.hidden_states
            if return_attn_weights:
                all_attentions = all_attentions + (layer_outputs.attention_weights,)
            if use_cache:
                all_past_key_values = all_past_key_values + (
                    layer_outputs.past_key_values,
                )

        if return_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return TransformerDecoderOutput(
            last_hidden_states=hidden_states,
            hidden_states=all_hidden_states,
            attention_weights=all_attentions,
            past_key_values=all_past_key_values,
        )


class SiLU(nn.Module):
    r"""Sigmoind Linear Unit

    .. math:: \text{SiLU}(x) = x * \sigma(1.702 * x)

    where :math:`\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class TransformerDecoderLayer(nn.Module):
    """A single layer from a transformer decoder

        `"On Layer Normalization in the Transformer Architecture"<https://arxiv.org/pdf/2002.04745.pdf>`_
        `"Improving Language Understanding by Generative
        Pre-Training"<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_

    Args:
        d_model (int):
    """

    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = SiLU,
        attn_module: nn.Module = FullAttention(attn_dropout=0.1),  # TODO: SelfAttention
    ) -> None:
        super().__init__()

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            n_layer=1,  # TODO: dummy
            attn_module=attn_module,
        )

        self.mlp = MLP(
            in_dim=d_model,
            out_dim=d_model,
            hidden_dims=[d_model * 4],
            dropout=0.0,
            activation=activation,
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        return_attn_weights: bool = False,
    ) -> TransformerLayerOutput:
        attn_probs = None
        past_key_values = None

        # Add head dim to attention mask for broadcasting
        if attn_mask is not None:
            attn_mask = get_extended_attention_mask(attn_mask)

        attn_out = self._attn(
            self.norm_attn(x),
            attn_mask,
            head_mask,
            return_attn_weights,
            use_cache=use_cache,
            causal=causal,
        )

        if return_attn_weights:
            x, attn_probs = attn_out
        else:
            x = attn_out

        if use_cache:
            past_key_values = self.attention.cache

        x = x + self.dropout_attn(x)
        x = self._mlp_block(self.norm_mlp(x))
        x = x + self.dropout_mlp(x)

        return TransformerLayerOutput(
            hidden_states=x,
            attention_weights=attn_probs,
            past_key_values=past_key_values,
        )

    @checkpoint_wrapper
    def _attn(
        self,
        x: Tensor,
        attn_mask: Tensor,
        head_mask: Tensor,
        return_attn_weights: bool,
        use_cache: bool,
        causal: bool,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return self.attention(
            x, x, x, attn_mask, head_mask, return_attn_weights, use_cache, causal
        )

    @checkpoint_wrapper
    def _mlp_block(self, x: Tensor) -> Tensor:
        return self.mlp(x)


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


def _get_clones(module: nn.Module, n: int) -> nn.Module:
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])
