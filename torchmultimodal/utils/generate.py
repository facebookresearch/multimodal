# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchmultimodal.utils.attention import get_causal_attention_mask


class Generation:
    """Utility class that generates sequence of one modality given prompt from another.

    Attributes:
        max_seq_len (int): Maximum length of the sequence to generate. For high dimensional data
            this should be equivalent to the length of the flattened encoded sequence (or latent
            sequence length).
        model (nn.Module): Model that performs the core generation functionalities.
            Please refer to :class:`~torchmultimodal.models.gpt.MultimodalGPT`
            for more details about the required attributes and methods.
        num_in_tokens (int): Number of input modality tokens.
        num_out_tokens (int): Number of output modality tokens.
    """

    def __init__(
        self,
        max_seq_len: int,
        model: nn.Module,
    ) -> None:
        if model.training:
            model = model.eval()
            warnings.warn(f"{type(model)} is now switched to 'eval' mode.")

        self.max_seq_len = max_seq_len
        self.model = model
        self.num_in_tokens = model.num_in_tokens
        self.num_out_tokens = model.num_out_tokens

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        use_cache: bool = True,
        causal: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Generates samples of the output modality based on multinomial distribution.

        Args:
            x (Tensor): Tensor of batched input, i.e., prompt for the generation.
            use_cache (bool): If ``True``, key/values of the attention layers will be cached to
                speed up generation. Defaults to ``True``.
            causal (bool): If ``True``, use causal attention. Defaults to ``False``.
            top_k (int, optional): Number of tokens with the highest probability to keep.
                Defaults to ``None``.
            top_p (int, optional): Threshold that determines the top tokens to keep in terms of
                cumulative probability.
                See py:func:`~torchmultimodal.utils.generate.top_k_top_p_filtering`. Defaults to
                ``None``.

        Returns:
            A tensor of generated output modality.
        """
        in_tokens = self.model.encode(x, "in", **kwargs)  # type: ignore
        batch_size, in_seq_len = in_tokens.shape
        total_seq_len = in_seq_len + self.max_seq_len
        attn_mask = get_causal_attention_mask(in_seq_len)  # (in_seq_len, in_seq_len)
        logits_mask = get_logits_mask(
            in_seq_len, self.max_seq_len, self.num_in_tokens, self.num_out_tokens  # type: ignore
        )  # (total_seq_len, num_total_tokens)

        # Feed the input modality tokens `(b, in_seq_len)` through the model's transformer to learn
        # the intermediate context vectors (i.e., key/value).
        # The sequence is shifted to the right by one unit so that the predicted token at each location
        # along the sequence is based off the previous token.
        # Note that the first token is predicted from the learnt start-of-sentence ("sos") token which
        # gets prepended to the sequence after the position embedding layer.
        # Attention mask is required to avoid attending to future positions along the sequence.
        # See :class:`~torchmultimodal.models.gpt.RightShift` for more implementation details.
        self.model.fwd(  # type: ignore
            in_tokens=in_tokens,
            attn_mask=attn_mask,
            use_cache=use_cache,
            causal=causal,
            right_shift=True,
        )

        # Take the last token of the input modality as the "sos" token for the ouput modality
        samples = [in_tokens[:, -1:]]
        idx = 0
        # Attention mask is not required as the cached key/value sequence is only up to the
        # current `decode_step`
        while idx < self.max_seq_len:
            # decode_step is w.r.t. the total sequence concat from in and out
            # Shift by in_seq_len because we start from the out sequence
            decode_step = idx + in_seq_len
            if idx == 0:
                out = self.model(
                    in_tokens=samples[-1],
                    in_pos_ids=torch.tensor([decode_step - 1]).unsqueeze(0),
                    logits_mask=logits_mask[[decode_step]],
                    use_cache=use_cache,
                    causal=causal,
                    right_shift=False,
                )
            else:
                out = self.model(
                    out_tokens=samples[-1],
                    out_pos_ids=torch.tensor([idx - 1]).unsqueeze(0),
                    logits_mask=logits_mask[[decode_step]],
                    use_cache=use_cache,
                    causal=causal,
                    right_shift=False,
                )

            assert hasattr(out, "logits"), f"{type(out)} does not have field 'logits'"
            logits = out.logits

            logits_view = logits.view(-1, logits.shape[-1])  # (b, num_tokens)
            if top_k is not None or top_p is not None:
                logits_view = top_k_top_p_filtering(
                    logits_view, top_k=top_k, top_p=top_p
                )
            probs = F.softmax(logits_view, dim=-1)
            samples.append(torch.multinomial(probs, 1))  # (b, 1)
            idx += 1

        samples = torch.cat(
            samples[1:], dim=1
        )  # remove the "sos" token: (b, out_seq_len)
        samples = self.model.decode(samples)  # type: ignore

        return samples


def get_logits_mask(
    in_seq_len: int, out_seq_len: int, num_in_tokens: int, num_out_tokens: int
) -> Tensor:
    """
    Generates masks to restrict prediction to be made from tokens of the same modality.
    """
    total_seq_len = in_seq_len + out_seq_len
    num_tokens = num_in_tokens + num_out_tokens
    logits_mask = torch.ones(total_seq_len, num_tokens)
    logits_mask[in_seq_len:, :num_in_tokens] = 0
    logits_mask[:in_seq_len, num_in_tokens:] = 0

    return logits_mask  # (total_seq_len, total_num_tokens)


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filters a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits (Tensor): Logits distribution shape ``(b, num_tokens)`` where ``b`` is batch size, ``num_tokens``
            is the number of tokens.
        top_k (int): Keeps the top-k tokens with the highest probability (top-k filtering).
        top_p (float): Keeps the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        filter_value (float): Constant value to filter unwanted logits. Defaults to ``-inf``.
        min_tokens_to_keep (int): Minimum number of tokens to keep per batch example in the output.

    Code reference: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Raises:
        ValueError: If 'top_k' or 'top_p' are outside of valid numerical ranges.
    """
    if top_k < 0:
        raise ValueError(f"'top-k' must be non-negative but got {top_k}.")
    if top_p > 1.0 or top_p < 0.0:
        raise ValueError(f"'top-p' must be within [0.0, 1.0] but got {top_p}.")

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits
