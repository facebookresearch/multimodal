# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, List, NamedTuple, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchmultimodal.utils.attention import get_causal_attention_mask


class SampleOutput(NamedTuple):
    """Outputs from :meth:`~torchmultimodal.utils.generate.GenerationUtil.sample`.

    Attributes:
        decoded (Any): Generated sample data for the ouput modality.
        tokens (Tensor): Generated tokens ``(b, seq_len)`` for the output modality before being decoded
            back to data.
        model_outputs (Tuple[Any, ...]): A tuple of length ``seq_len`` containing output objects from
            the model's forward pass at each step of generation.
    """

    decoded: Any
    tokens: Tensor
    model_outputs: Tuple[Any, ...]


class GenerationUtil:
    """Utility class containing functions for multimodal auto-regressive generation.

    This class wraps around a ``nn.Module`` to generate data of one modality given
    inputs from another. While being agnostic to the architecture of the wrapped model,
    the latter needs to implement APIs to:
        * encode/decode between data and token representations
        * look up embeddings given token ids
        * compute scores for prediction
    See :class:`~torchmultimodal.models.gpt.MultimodalGPT` for the API details.

    Args:
        model (nn.Module): Model that is wrapped for generation.

    Attributes:
        num_in_tokens (int): Number of unique token states for the input modality.
        num_out_tokens (int): Number of unique token states for the output modality.
    """

    def __init__(self, model: nn.Module) -> None:
        if model.training:
            model = model.eval()
            warnings.warn(f"{type(model)} is now switched to 'eval' mode.")

        self.model = model
        self.num_in_tokens = model.num_in_tokens
        self.num_out_tokens = model.num_out_tokens

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        max_seq_len: int,
        use_cache: bool = True,
        causal: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_attn_weights: bool = False,
        return_hidden_states: bool = False,
        **model_kwargs: Any,
    ) -> SampleOutput:
        """Generates samples of the output modality based on multinomial distribution.

        Args:
            x (Tensor): Tensor of batched input, i.e., prompt for the generation.
            max_seq_len (int): Maximum length of the sequence to generate. For high dimensional data
                this should be equivalent to the length of the flattened encoded sequence.
            use_cache (bool, optional): If ``True``, key/values of the attention layers will be cached to
                speed up generation. Defaults to ``True``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            top_k (int, optional): Number of tokens with the highest probability to keep.
                Defaults to ``None``.
            top_p (float, optional): Threshold that determines the top tokens to keep in terms of
                cumulative probability. Defaults to ``None``.
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.
            model_kwargs (Any): Additional model specific kwargs will be forwarded to the ``forward``
                function of the model.

        Returns:
            An instance of :class:`~torchmultimodal.utils.generate.SampleOutput`.
        """
        in_tokens = self.model.encode(x, "in", **model_kwargs)  # type: ignore
        batch_size, in_seq_len = in_tokens.shape
        attn_mask = get_causal_attention_mask(in_seq_len)  # (in_seq_len, in_seq_len)
        # Construct step-wise logits mask
        logits_mask = get_logits_mask(
            in_seq_len=0,
            num_in_tokens=self.num_in_tokens,  # type: ignore
            out_seq_len=1,
            num_out_tokens=self.num_out_tokens,  # type: ignore
        )

        # Feed the input modality tokens `(b, in_seq_len)` through the model's transformer to learn
        # the intermediate context vectors (i.e., key/value).
        # The sequence is shifted to the right by one unit so that the predicted token at each location
        # along the sequence is based off the previous token.
        # Note that the first token is predicted from the learnt start-of-sentence ("sos") token which
        # gets prepended to the sequence after the position embedding layer.
        # Attention mask is required to avoid attending to future positions along the sequence.
        # See :class:`~torchmultimodal.models.gpt.RightShift` for more implementation details.
        _ = self.model.fwd(  # type: ignore
            in_tokens=in_tokens,
            attn_mask=attn_mask,
            use_cache=use_cache,
            causal=causal,
            right_shift=True,
            return_attn_weights=return_attn_weights,
            return_hidden_states=return_hidden_states,
        )

        model_outputs: Tuple[Any, ...] = ()
        samples: List[Tensor] = []
        idx = 0
        while idx < max_seq_len:
            # Attention mask is not required as the cached key/value sequence is only up to the
            # current step
            if idx == 0:
                # Take the last token of the input modality as the "sos" token for the ouput modality
                out = self.model(
                    in_tokens=in_tokens[:, -1:],
                    in_pos_ids=torch.tensor([in_seq_len - 1]).unsqueeze(0),
                    logits_mask=logits_mask,
                    use_cache=use_cache,
                    causal=causal,
                    right_shift=False,
                    return_attn_weights=return_attn_weights,
                    return_hidden_states=return_hidden_states,
                )
            else:
                out = self.model(
                    out_tokens=samples[-1],
                    out_pos_ids=torch.tensor([idx - 1]).unsqueeze(0),
                    logits_mask=logits_mask,
                    use_cache=use_cache,
                    causal=causal,
                    right_shift=False,
                    return_attn_weights=return_attn_weights,
                    return_hidden_states=return_hidden_states,
                )

            assert hasattr(out, "logits"), f"{type(out)} does not have field 'logits'"
            logits = out.logits

            logits_view = logits.view(-1, logits.shape[-1])  # (b, num_tokens)
            logits_view = self._filter_logits(logits_view, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits_view, dim=-1)
            samples.append(torch.multinomial(probs, 1) - self.num_in_tokens)  # (b, 1)
            model_outputs = model_outputs + (out,)
            idx += 1

        samples = torch.cat(samples, dim=1)
        decoded = self.model.decode(samples)  # type: ignore

        return SampleOutput(
            decoded=decoded, tokens=samples, model_outputs=model_outputs
        )

    def _filter_logits(
        self, logits: Tensor, top_k: Optional[int] = None, top_p: Optional[float] = None
    ) -> Tensor:
        logits_filters: List[Any] = []
        if top_k is not None:
            logits_filters.append(LogitsFilterTopK(top_k))
        if top_p is not None:
            logits_filters.append(LogitsFilterTopP(top_p))

        for _filter in logits_filters:
            logits = _filter(logits)

        return logits


def get_logits_mask(
    in_seq_len: int = 0,
    out_seq_len: int = 0,
    num_in_tokens: int = 0,
    num_out_tokens: int = 0,
) -> Tensor:
    """Applies masks to logits to restrict prediction from being made of tokens of the opposite modality.

    Args:
        in_seq_len (int, optional): Length of input modality sequence from the logits tensor. Defaults to ``0``.
        out_seq_len (int, optional): Length of output modality sequence from the logits tensor.
            Defaults to ``0``.
        num_in_tokens (int, optional): Number of input modality token states from the model. Defaults to ``0``.
        num_out_tokens (int, optional): Number of output modality token states from the model.
            Defaults to ``0``.

    Returns:
        Logits mask tensor containing ``1``s for unmasked positions and ``0``s for masked ones.
    """
    mask = torch.zeros(in_seq_len + out_seq_len, num_in_tokens + num_out_tokens)
    # the quadrant of input modality sequence, input token states should not be masked
    # Similar for that of output modality sequence, output token states
    mask[in_seq_len:, num_in_tokens:] = 1
    mask[:in_seq_len, :num_in_tokens] = 1

    return mask


class LogitsFilterTopK:
    """Filters a distribution of logits using top_k

    Code reference: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
        top_k (int, optional): Keeps the top_k tokens with the highest probability (top_k filtering).
            Defaults to ``None``.
        filter_value (float, optional): Constant value to filter unwanted logits. Defaults to ``-inf``.
        min_tokens_to_keep (int, optional): Minimum number of tokens to keep per batch example in the output.
            Defaults to ``1``.

    Raises:
        ValueError: If 'top_k' is outside of valid numerical ranges.

    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        min_tokens_to_keep: int = 1,
        filter_value: float = -float("inf"),
    ) -> None:
        if top_k is not None and top_k < 0:
            raise ValueError(f"'top_k' must be non-negative but got {top_k}.")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.filter_value = filter_value
        self.top_k = top_k

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Args:
            logits (Tensor): Logits distribution shape ``(b, num_tokens)`` where ``b`` is batch size,
                ``num_tokens`` is the number of tokens.

        Returns:
            Filtered logits tensor.
        """
        if self.top_k == 0:
            return logits

        top_k = min(
            max(self.top_k, self.min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]  # (b, 1)
        logits[indices_to_remove] = self.filter_value

        return logits


class LogitsFilterTopP:
    """Filters a distribution of logits using nucleus (top_p) filtering

    Code reference: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Args:
        top_p (float, optional): Keeps the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).
            Defaults to ``None``.
        filter_value (float, optional): Constant value to filter unwanted logits. Defaults to ``-inf``.
        min_tokens_to_keep (int, optional): Minimum number of tokens to keep per batch example in the output.
            Defaults to ``1``.

    Raises:
        ValueError: If 'top_p' is outside of valid numerical ranges.
    """

    def __init__(
        self,
        top_p: Optional[float] = None,
        min_tokens_to_keep: int = 1,
        filter_value: float = -float("inf"),
    ) -> None:
        if top_p is not None and (top_p > 1.0 or top_p < 0.0):
            raise ValueError(f"'top_p' must be within `[0.0, 1.0]` but got {top_p}.")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.filter_value = filter_value
        self.top_p = top_p

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Args:
            logits (Tensor): Logits distribution shape ``(b, num_tokens)`` where ``b`` is batch size,
                ``num_tokens`` is the number of tokens.

        Returns:
            Filtered logits tensor.
        """
        if self.top_p == 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add
            # the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = self.filter_value

        return logits
