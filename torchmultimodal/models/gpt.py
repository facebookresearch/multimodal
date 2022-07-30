# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.attention import MultiHeadAttention, SelfAttention
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.utils.attention import get_extended_attention_mask
from torchmultimodal.utils.common import checkpoint_wrapper, get_clones


class TransformerDecoderOutput(NamedTuple):
    last_hidden_states: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attention_weights: Optional[Tuple[Tensor, ...]] = None
    past_key_values: Optional[Tuple[Dict[str, Tensor], ...]] = None


class TransformerLayerOutput(NamedTuple):
    hidden_states: Tensor
    attention_weights: Optional[Tensor] = None
    past_key_values: Optional[Dict[str, Tensor]] = None


class MultimodalTransformerDecoder(nn.Module):
    """Extends the transformer decoder of GPT (Generative Pre-Training) model for cross-modality generation.

    This module implements the transformer decoder of GPT model for generation of one modality given another
    following the paper `"Improving Language Understanding by Generative Pre-Training
    "<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_.
    The position embedding layers are per modality:
        * During training both modalities are fed into the module and concatenated as a single sequence of
            tokenized embedding vectors
        * During generation the future data points are predicted step-wise from the past. The input modality
            is processed before the output modality (see ``torchmultimodal.utils.common.generate``). Therefore,
            at any point in time the input data contains only one modality.

    Attributes:
        in_pos_emb (nn.Module): input modality position embedding layer.
        out_pos_emb (nn.Module): output modality position embedding layer.
        decoder (nn.Module): the transformer decoder (see ``torchmultimodal.models.gpt.TransformerDecoder``)

    Args:
        in_modality (Tensor, optional): Tensor of dimension ``(b, in_seq_len, c)`` containing tokenized
            embeddings for the input modality. Defaults to ``None``.
        out_modality (Tensor, optional): Tensor of dimension ``(b, out_seq_len, c')`` containing tokenized
            embeddings for the output modality. Defaults to ``None``.
        in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
            input modality position embeddings. Defaults to ``None``.
        out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
            output modality position embeddings. Defaults to ``None``.
        attn_mask (Tensor, optional): Tensor of dimension ``(b, out_seq_len, in_seq_len)``. Contains 1s for
            positions to attend to and 0s for masked positions. Defaults to ``None``.
        head_mask (Tensor, optional): Tensor of dimension ``(b, h, out_seq_len, in_seq_len)``. Contains 1s
            for attention heads to use and 0s for masked heads. Defaults to ``None``.
        use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
            recomputes key and value for each decoding step. Defaults to ``False``.
        causal (bool. optional): If ``True``, use causal attention. Defaults to ``False``.
        return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
            layer. Defaults to ``False``.
        return_hidden_states (bool): If ``True``, returns the embeddings of each transformer layer. Defaults to
            ``False``.
    """

    def __init__(
        self,
        in_pos_emb: nn.Module,
        out_pos_emb: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()

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
    ) -> TransformerDecoderOutput:
        if (in_modality is None) and (out_modality is None):
            raise ValueError(
                "in_modality and out_modality sequences cannot be both empty"
            )

        # Since generation is based on the previous data point (autoregressive) where
        # only one modality is needed at any point along the sequence, either input
        # or output modality can be None.
        # Whereas training is done by paralleling all data points so both modalities
        # should be present. Position ids are optional as they can be derived from
        # the sequence length of each modality.
        if in_modality is None:
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x = out_modality + self.out_pos_emb(out_pos_ids)
        elif out_modality is None:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            x = in_modality + self.in_pos_emb(in_pos_ids)
        else:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x_in = in_modality + self.in_pos_emb(in_pos_ids)
            x_out = out_modality + self.out_pos_emb(out_pos_ids)
            x = torch.cat((x_in, x_out), dim=1)

        return self.decoder(
            x,
            attn_mask,
            head_mask,
            use_cache,
            causal,
            return_attn_weights,
            return_hidden_states,
        )

    def _norm_pos_ids(self, x: Tensor, pos_ids: Optional[Tensor] = None) -> Tensor:
        b, seq_len, _ = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)[
                None, :
            ]  # (b, seq_len)

        if pos_ids.shape != (b, seq_len):
            raise ValueError(
                "input sequence and position ids must be equal in batch size and length"
            )

        return pos_ids


class TransformerDecoder(nn.Module):
    """A transformer decoder

    Attributes:
        decoder_layer (nn.Module): The transformer decoder layer.
        num_layers (int): The number of transformer decoder layers to be stacked up.

    Args:
        hidden_states (Tensor): Tensor of the embedding vectors of dimension ``(b, seq_len, emb_dim)``.
        attn_mask (Tensor, optional): Tensor of dimension ``(b, out_seq_len, in_seq_len)``. Contains 1s for
            positions to attend to and 0s for masked positions. Defaults to ``None``.
        head_mask (Tensor, optional): Tensor of dimension ``(b, h, out_seq_len, in_seq_len)``. Contains 1s
            for attention heads to use and 0s for masked heads. Defaults to ``None``.
        use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
            recomputes key and value for each decoding step. Defaults to ``False``.
        causal (bool. optional): If ``True``, use causal attention. Defaults to ``False``.
        return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
            layer. Defaults to ``False``.
        return_hidden_states (bool): If ``True``, returns the embeddings of each transformer layer. Defaults to
            ``False``.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
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
        all_hidden_states: Tuple[Tensor, ...] = () if return_hidden_states else None
        all_attentions: Tuple[Tensor, ...] = () if return_attn_weights else None
        all_past_key_values: Tuple[Dict[str, Tensor], ...] = () if use_cache else None

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
    r"""Sigmoid Linear Unit

    .. math:: \text{SiLU}(x) = x * \sigma(1.702 * x)

    where :math:`\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(1.702 * x) * x


class TransformerDecoderLayer(nn.Module):
    """A single layer from a GPT transformer decoder

    Layer norm is applied before the attention layer and the feedforward layer so that the gradients are
    well-behaved at initialization for training stability. This is also called "Pre-LN Transformer" studied in
    `"On Layer Normalization in the Transformer Architecture"<https://arxiv.org/pdf/2002.04745.pdf>`_

    Attributes:
        d_model (int): Dimension of the input embedding vector.
        n_head (int): Number of attention heads.
        dropout (float, optional): Dropout probability used in the dropout layers. Defaults to ``0.1``.
        activation (Union[str, Callable], optional): Activation used by the feedforward layer. Defaults to
            ``SiLU``.
        attn_module (nn.Module): Self attention module. Defaults to ``SelfAttention`` with dropout rate equal
            to ``0.1``.

    Args:
        x (Tensor): input embedding vectors.
        attn_mask (Tensor, optional): Tensor of dimension ``(b, out_seq_len, in_seq_len)``. Contains 1s for
            positions to attend to and 0s for masked positions. Defaults to ``None``.
        head_mask (Tensor, optional): Tensor of dimension ``(b, h, out_seq_len, in_seq_len)``. Contains 1s
            for attention heads to use and 0s for masked heads. Defaults to ``None``.
        use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
            recomputes key and value for each decoding step. Defaults to ``False``.
        causal (bool. optional): If ``True``, use causal attention. Defaults to ``False``.
        return_attn_weights (bool, optional): If ``True``, returns attention probabilities of the layer.
            Defaults to ``False``.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_head: int = 12,
        dropout: float = 0.1,
        activation: Callable[..., nn.Module] = SiLU,
        attn_module: nn.Module = SelfAttention(attn_dropout=0.1),
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
            x,
            attention_mask=attn_mask,
            head_mask=head_mask,
            return_attn_weights=return_attn_weights,
            use_cache=use_cache,
            causal=causal,
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

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(torch.FloatTensor(embedding_dim).normal_(std=0.02))

    def forward(self, x: Tensor) -> Tensor:
        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2)  # (batch, seq_len, emb)
        sos = self.sos.unsqueeze(0).unsqueeze(1).repeat(x_shape[0], 1, 1)  # (b, 1, emb)
        # Shift one unit to the right along dim ``seq_len``
        x = torch.cat(
            (sos.data, x[:, :-1, :]), dim=1
        )  # (batch, seq_len, embedding_dim)
        x = x.view(*x_shape)
        return x
