# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from torchmultimodal.modules.layers.activation import SiLU
from torchmultimodal.modules.layers.attention import MultiHeadAttention, SelfAttention
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.utils.common import checkpoint_wrapper, get_clones


class TransformerDecoderOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.gpt.TransformerDecoder`.

    Attributes:
        last_hidden_states (Tensor): Output from the last layer of the transformer.
        hidden_states (Tuple[Tensor, ...], optional): Outputs from all layers of the transformer.
            Defaults to ``None``.
        attention_weights (Tuple[Tensor, ...], optional): Attention probabilities from all layers of the
            transformer. Defaults to ``None``.
        past_key_values (Tuple[Dict[str, Tensor], ...]], optional): If ``use_cache`` is on, contains
            key/value tensors prior to the current step along the sequence. Defaults to ``None``.
    """

    last_hidden_states: Tensor
    hidden_states: Optional[Tuple[Tensor, ...]] = None
    attention_weights: Optional[Tuple[Tensor, ...]] = None
    past_key_values: Optional[Tuple[Dict[str, Tensor], ...]] = None


class TransformerLayerOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.gpt.TransformerDecoderLayer`.

    Attributes:
        hidden_states (Tensor): Output from the current layer.
        attention_weights (Tensor, optional): Attention probability tensor of the current layer.
            Defaults to ``None``.
        past_key_values (Dict[str, Tensor], optional): If ``use_cache`` is on, contains key/value tensors
            prior to the current step along the sequence. Defaults to ``None``.
    """

    hidden_states: Tensor
    attention_weights: Optional[Tensor] = None
    past_key_values: Optional[Dict[str, Tensor]] = None


class MultimodalGPTOutput(NamedTuple):
    """Outputs from :meth:`~torchmultimodal.models.gpt.MultimodalGPT.forward`.

    Attributes:
        decoder_output (TransformerDeocoderOutput): Contains output from the multimodal transformer decoder.
            See :class:`MultimodalTransformerDecoder`.
        logits (Tensor): Logits computed from the last hidden state of the multimodal transformer decoder.
    """

    decoder_output: TransformerDecoderOutput
    logits: Tensor


class MultimodalGPT(nn.Module):
    """Extends the GPT (Generative Pre-Training) model for cross-modality generation.

    This module implements the GPT model for generation of one modality given another
    following the paper `"Improving Language Understanding by Generative Pre-Training
    "<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_.

    Args:
        d_model (int): Embedding dimension of the transformer decoder.
        num_in_tokens (int): Number of unique token states for the input modality.
        num_out_tokens (int): Number of unique token states for the output modality.
        latent_shape ([Tuple[int, ...]): Shape of the latent space of the output modality tokenizer. Used to reshape
            sequence of generated tokens to be decoded back to data.
        in_tokenizer (nn.Module): Tokenizer for the input modality. Must have methods ``encode``, ``lookup``.
        out_tokenizer (nn.Module): Tokenizer for the output modality. Must have methods ``encode``, ``decode``.
        mm_decoder (nn.Module): Multimodal transformer decoder. An instace of
            :py:class:`MultimodalTransformerDecoder`.
        in_projection (nn.Module, optional): Projects the input modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        out_projection (nn.Module, optional): Projects the output modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        norm_layer (Callable[..., nn.Module], optional): Which normalization layer to use. Supports ``nn.Module`` or
            partial. If ``None``, ``nn.LayerNorm`` will be used as the default.
        use_gpt_init (bool): Whether to use GPT model specific initialization. Defaults to ``True``.

    Raises:
        AttributeError: If input tokenizer does not implement methods ``encode`` and ``lookup`` or if output
        tokenizer does not implement methods ``encode``, ``lookup`` and ``decode``.
    """

    def __init__(
        self,
        d_model: int,
        num_in_tokens: int,
        num_out_tokens: int,
        latent_shape: Tuple[int, ...],
        in_tokenizer: nn.Module,
        out_tokenizer: nn.Module,
        mm_decoder: nn.Module,
        in_projection: Optional[nn.Module] = None,
        out_projection: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_gpt_init: bool = True,
    ) -> None:
        super().__init__()
        if not all(
            [hasattr(in_tokenizer, attr_name) for attr_name in ["encode", "lookup"]]
        ):
            raise AttributeError(
                "Input modality tokenizer must have methods 'encode' and 'lookup'."
            )

        if not all(
            [
                hasattr(out_tokenizer, attr_name)
                for attr_name in ["encode", "lookup", "decode"]
            ]
        ):
            raise AttributeError(
                "Output modality tokenizer must have methods 'encode', 'lookup' and 'decode'."
            )

        num_tokens = num_in_tokens + num_out_tokens
        self.num_in_tokens = num_in_tokens
        self.num_out_tokens = num_out_tokens
        self.latent_shape = latent_shape
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.mm_decoder = mm_decoder
        self.in_projection = in_projection
        self.out_projection = out_projection
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.norm = norm_layer(normalized_shape=d_model)
        self.to_logit = nn.Linear(d_model, num_tokens, bias=False)
        # This will give us equal probabilities after the soft max layer initially to avoid biasing
        # towards any particular prediction category
        self.to_logit.weight.data.copy_(torch.zeros(num_tokens, d_model))

        if use_gpt_init:
            self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Initialize weights of the layers in question, e.g.,  after loading checkpoints
        # Only do this when the layers have weights data, e.g., for text tokenizer the projection
        # layer is dummy (nn.Identity)
        if hasattr(self.in_projection, "weight"):
            self.in_projection.weight.data.normal_(std=0.02)  # type: ignore
        if hasattr(self.out_projection, "weight"):
            self.out_projection.weight.data.normal_(std=0.02)  # type: ignore

    def forward(
        self,
        in_tokens: Optional[Tensor] = None,
        out_tokens: Optional[Tensor] = None,
        in_pos_ids: Optional[Tensor] = None,
        out_pos_ids: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        logits_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        right_shift: bool = False,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> MultimodalGPTOutput:
        """
        Args:
            in_tokens (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing tokens
                for the input modality. Defaults to ``None``.
            out_tokens (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing tokens
                for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            logits_mask (Tensor, optional): Tensor of dimension ``(seq_len, num_tokens)`` or
                ``(b, seq_len, num_tokens)`` to ensure we only calculate probabilities from tokens of the
                corresponding modality sequence.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
                recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instance of :class:`~torchmultimodal.models.gpt.MultimodalGPTOutput`.
        """
        decoder_output = self.fwd(
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            in_pos_ids=in_pos_ids,
            out_pos_ids=out_pos_ids,
            attn_mask=attn_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            causal=causal,
            right_shift=right_shift,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
        )

        hidden_states = decoder_output.last_hidden_states
        logits = self.logit_projection(hidden_states, logits_mask)

        return MultimodalGPTOutput(decoder_output, logits)

    def fwd(
        self,
        in_tokens: Optional[Tensor] = None,
        out_tokens: Optional[Tensor] = None,
        in_pos_ids: Optional[Tensor] = None,
        out_pos_ids: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        causal: bool = False,
        right_shift: bool = False,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerDecoderOutput:
        # During training this method is used in the forward pass to decode input- and
        # output- tokens.
        # During generation this method is used for autoregressive decoding.
        if (in_tokens is None) and (out_tokens is None):
            raise ValueError(
                "input-modality token and output-modality token sequences cannot be both empty"
            )

        # Look up embeddings for the given tokens and project to fit the size of the
        # transformer decoder
        in_modality = out_modality = None

        if in_tokens is not None:
            # (b, in_seq_len, in_emb_dim)
            in_modality = self.lookup(in_tokens, "in")
            if self.in_projection is not None:
                in_modality = self.in_projection(
                    in_modality
                )  # (b, in_seq_len, d_model)
        if out_tokens is not None:
            # (b, out_seq_len, out_emb_dim)
            out_modality = self.lookup(out_tokens, "out")
            if self.out_projection is not None:
                out_modality = self.out_projection(
                    out_modality
                )  # (b, out_seq_len, d_model)

        return self.mm_decoder(
            in_modality=in_modality,
            out_modality=out_modality,
            in_pos_ids=in_pos_ids,
            out_pos_ids=out_pos_ids,
            attn_mask=attn_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            causal=causal,
            right_shift=right_shift,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
        )

    def logit_projection(
        self, hidden_states: Tensor, logits_mask: Optional[Tensor] = None
    ) -> Tensor:
        if logits_mask is not None and logits_mask.dim() == 2:
            logits_mask = logits_mask.unsqueeze(
                0
            )  # (seq_len, num_tokens) -> (1, seq_len, num_tokens)

        hidden_states = self.norm(hidden_states)
        logits = self.to_logit(hidden_states)
        max_neg_value = -torch.finfo(logits.dtype).max
        if logits_mask is not None:
            logits.masked_fill_(logits_mask == 0, max_neg_value)

        return logits  # (b, seq_len, num_tokens)

    def encode(self, x: Any, modality: str, **kwargs: Any) -> Tensor:
        """Converts data to token ids.

        Although this is not part of the forward pass, it is used to generate labels for training
        as well as inputs for autoregressive decoding.

        Args:
            x (Any): Data to be encoded, e.g., ``List[str]`` for text, ``Tensor`` of shape
                ``(b, c, d1, ..., dn)`` for audio/image/video.
            modality (str): Input or output modality string used to select the encoder.
            kwargs (Any): Other keyword arguments suitable for the encoder.

        Returns:
            A tensor of token ids of shape ``(b, seq_len)``.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == "in":
            encoder = self.in_tokenizer.encode
        elif modality == "out":
            encoder = self.out_tokenizer.encode
        else:
            raise ValueError(f"Invalid modality parameter: {modality}")

        token_ids = encoder(x, **kwargs)  # type: ignore

        # For generation we need to flatten the tokens
        return token_ids.flatten(
            start_dim=1, end_dim=-1
        )  # (b, d1, ..., dn) -> (b, seq_len)

    def decode(self, token_ids: Tensor, **kwargs: Any) -> Any:
        """Converts out-modality tokens ids back to data during generation.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)`` to be decoded.
            kwargs (Any): Other keywords arguments suitable for the decoder.

        Returns:
            The decoded data, e.g., ``List[str]`` for text, a tensor of shape ``(b, c, d1. ,,, dn)`` for
                audio/image/video.

        Raises:
            ValueError: If the shape of ``token_ids`` is not of dimension two.
            ValueError: If the sequence dim of ``token_ids`` does not match that inferred from ``latent_shape``.
        """
        if len(token_ids.shape) != 2:
            raise ValueError(
                f"Shape of token ids should be '(batch_size, sequence_length)' but got {token_ids.shape}"
            )
        # Check if the generated sequence length matches that inferred from the latent embedding space
        latent_seq_len = torch.prod(torch.tensor(self.latent_shape)).item()
        if token_ids.shape[1] != latent_seq_len:
            raise ValueError(
                f"Sequence to decode does not match that inferred from the tokenizer: {latent_seq_len}"
            )

        # Reshape the sequence of token ids back to dim of latent space
        token_ids = token_ids.view(
            token_ids.shape[0], *self.latent_shape
        )  # (b, seq_len) -> (b, d1, ..., dn)

        return self.out_tokenizer.decode(token_ids, **kwargs)  # type: ignore

    def lookup(self, token_ids: Tensor, modality: str) -> Tensor:
        """Looks up the latent embeddings corresponding to the token ids during generation.

        We ask each tokenizer to implement this method. An example is :class:`torchmultimodal.models.vqvae.VQVAE`.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)``.
            modality (str): The modality at which this method is performed.

        Returns:
            A tensor of embeddings corresponding to the token ids.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == "in":
            tokenizer = self.in_tokenizer
        elif modality == "out":
            tokenizer = self.out_tokenizer
        else:
            raise ValueError(f"Invalid modality parameter: {modality}")

        return tokenizer.lookup(token_ids)  # type: ignore


class MultimodalTransformerDecoder(nn.Module):
    """A transformer decoder for two modalities

    The token- and position- embedding layers are per modality:
        * During training both modalities are fed into the module and concatenated as a single sequence of
            tokenized embedding vectors
        * During generation the future data points are predicted step-wise from the past. The input modality
            is processed before the output modality (see ``torchmultimodal.utils.common.generate``). Therefore,
            at any point in time the input data contains only one modality.

    Args:
        in_pos_emb (nn.Module): Input modality position embedding layer.
        out_pos_emb (nn.Module): Output modality position embedding layer.
        decoder (nn.Module): The transformer decoder. An instance of :py:class:`TransformerDecoder`.
        right_shift (nn.Module): Layer that shifts the embedding vectors to the right and prepends it with
            start of sentence token (SOS). An instance of :py:class:`RightShift`.

    Note:
        * During training mode, the SOS token is prepended to the left of the concatenated input and
            output modality sequence;
        * During generation mode, the SOS token is only required for the input modality sequence as
            the initial token to be learnt from. Right shift should be turned off
            (``right_shift = False``, see args) when we start to generate the output modality samples.
    """

    def __init__(
        self,
        in_pos_emb: nn.Module,
        out_pos_emb: nn.Module,
        decoder: nn.Module,
        right_shift: nn.Module,
    ) -> None:
        super().__init__()

        self.in_pos_emb = in_pos_emb
        self.out_pos_emb = out_pos_emb
        self.decoder = decoder
        self.right_shift = right_shift

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
        right_shift: bool = False,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
    ) -> TransformerDecoderOutput:
        """
        Args:
            in_modality (Tensor, optional): Tensor of dimension ``(b, in_seq_len, d_model)`` containing tokenized
                embeddings for the input modality. Defaults to ``None``.
            out_modality (Tensor, optional): Tensor of dimension ``(b, out_seq_len, d_model')`` containing tokenized
                embeddings for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding.
                If ``False``, recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instace of :class:`~torchmultimodal.models.gpt.TransformerDecoderOutput`.
        """
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

        if self.training or right_shift:
            x = self.right_shift(x)

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
        _, seq_len, _ = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)[
                None, :
            ]  # (1, seq_len)

        if pos_ids.shape[1] != seq_len:
            raise ValueError(
                f"Input sequence and position ids must be equal in length: {pos_ids.shape[1]} != {seq_len}"
            )

        return pos_ids


class TransformerDecoder(nn.Module):
    """A transformer decoder.

    Args:
        decoder_layer (nn.Module): The transformer decoder layer.
            An instance of :class:`TransformerDecoderLayer`.
        num_layers (int): The number of transformer decoder layers to be stacked up.
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
        """
        Args:
            hidden_states (Tensor): Tensor of the embedding vectors of dimension ``(b, seq_len, emb_dim)``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
                recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instance of :class:`~torchmultimodal.models.gpt.TransformerDecoderOutput`.
        """
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask[
                None, None, :, :
            ]  # (q_seq_len, k_seq_len) -> (1, 1, q_seq_len, k_seq_len)

        if head_mask is not None and head_mask.dim() == 3:
            head_mask = head_mask[None, :, :, :]

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


class TransformerDecoderLayer(nn.Module):
    """A single layer from a GPT transformer decoder

    Layer norm is applied before the attention layer and the feedforward layer so that the gradients are
    well-behaved at initialization for training stability. This is also called "Pre-LN Transformer" studied in
    `"On Layer Normalization in the Transformer Architecture"<https://arxiv.org/pdf/2002.04745.pdf>`_

    Args:
        d_model (int): Dimension of the embeddings.
        n_head (int): Number of attention heads.
        dropout (float, optional): Dropout probability used in the dropout layers. Defaults to ``0.1``.
        activation (Union[str, Callable], optional): Activation used by the feedforward layer. Defaults to
            ``SiLU``.
        attn_module (nn.Module): Self attention module. Defaults to ``SelfAttention`` with dropout rate equal
            to ``0.1``.
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

        # No bias when projecting q, k, v in GPT model
        # https://github.com/openai/gpt-2/blob/master/src/model.py#L54
        self.attention = MultiHeadAttention(
            dim_q=d_model,
            dim_kv=d_model,
            n_head=n_head,
            attn_module=attn_module,
            add_bias=False,
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
        """
        Args:
            x (Tensor): input embedding vectors.
            attn_mask (Tensor, optional): Tensor of dimension ``(b, q_seq_len, k_seq_len)`` where prefixes ``q``
                and ``k`` stand for query and key. Contains 1s for positions to attend to and 0s for masked positions.
                Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(b, h, q_seq_len, k_seq_len)``. Masks need to be
                specified for each attention head. Defaults to ``None``.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
                recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of the layer.
                Defaults to ``False``.

        Returns:
            An instance of :class:`~torchmultimodal.models.gpt.TransformerLayerOutput`.
        """
        attn_probs = None
        past_key_values = None

        attn_out = self._attn(
            self.norm_attn(x),
            attn_mask,
            head_mask,
            return_attn_weights,
            use_cache=use_cache,
            causal=causal,
        )

        if return_attn_weights:
            attn_hidden_states, attn_probs = attn_out
        else:
            attn_hidden_states = attn_out

        if use_cache:
            past_key_values = self.attention.cache

        x = x + self.dropout_attn(attn_hidden_states)

        mlp_hidden_states = self._mlp_block(self.norm_mlp(x))
        x = x + self.dropout_mlp(mlp_hidden_states)

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
    """Shifts the embedding vectors along the sequence dimension to the right.

    Since the decoder progresses by taking the token it generates in the previous step, before it
    has generated anything it needs a token to start with. Hence, the start-of-sentence (SOS) token.
    The SOS token is a learnable parameter of the decoder and the choice of its initialization is taken
    from VideoGPT: https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py#L517

    Args:
        embedding_dim (int): Dimension of the embedding vector for each token along the sequence.

    Attributes:
        sos (nn.Parameter): The starting token to be prepended to the sequence.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(torch.FloatTensor(embedding_dim).normal_(std=0.02))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): An input tensor of shape ``(b, seq_len, emb_dim)``.

        Returns;
            A tensor of the same shape as that of the input with the ``sos`` token prepended.
        """
        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2)  # (batch, seq_len, emb)
        sos = self.sos.unsqueeze(0).unsqueeze(1).repeat(x_shape[0], 1, 1)  # (b, 1, emb)
        # Shift one unit to the right along dim ``seq_len``
        x = torch.cat(
            (sos.data, x[:, :-1, :]), dim=1
        )  # (batch, seq_len, embedding_dim)
        x = x.view(*x_shape)
        return x
