# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch

from examples.mugen.generation.video_vqvae import video_vqvae_mugen

from tokenizers import Tokenizer  # type: ignore
from torch import nn, Tensor
from torchmultimodal import _PATH_MANAGER

from torchmultimodal.models.gpt import (
    MultimodalGPT,
    MultimodalTransformerDecoder,
    RightShift,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torchmultimodal.modules.layers.attention import SelfAttention
from torchmultimodal.modules.layers.position_embedding import (
    BroadcastedPositionEmbedding,
)


BPE_PRETRAINED_TOKENIZER_URL = "https://pytorch.s3.amazonaws.com/models/multimodal/mugen/tokenizer-coinrun_1024.json"


# TODO: Load gpt ckpt + add option to builder to load from pretrained (will remove after
# the next PR)
def text_video_gpt(
    text_seq_len: int = 128,
    video_seq_len: int = 32,
    resolution: int = 256,
    downsample: Tuple[int, int, int] = (4, 32, 32),
    d_model: int = 768,
    n_head: int = 8,
    dropout: float = 0.2,
    attn_dropout: float = 0.3,
    num_decoder_layers: int = 12,
    use_gpt_init: bool = True,
    text_pretrained_tokenizer_url: str = BPE_PRETRAINED_TOKENIZER_URL,
    video_vqvae_pretrained_model_key: Optional[str] = None,
) -> MultimodalGPT:
    """Builds a text-to-video GPT model from user inputs

    Parameter defaults follow MUGEN project:
        * Video VQVAE: https://github.com/mugen-org/MUGEN_baseline/tree/main/generation/experiments/vqvae
        * GPT: https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/gpt/gpt.py#L252

    Args:
        text_seq_len (int): Length of text sequences after padding. Defaults to ``128``.
        video_seq_len (int): Length of video sequences sampled from the dataset. Defaults to ``32``. Other
            values used by MUGEN are ``8``, ``16``.
        resolution (int): Resolution of the sampled video sequences defining height and width of each frame.
            Defaults to ``256``.
        downsample (Tuple[int, int, int]): Ratio by which to disperse along each dimension the sampled sequences.
            For example, if the original frame is ``(32, 256, 256)``, after downsampling by ``(4, 32, 32)`` the
            new frame will be of shape ``(8, 8, 8)`` with each dim divided by the rate of downsample. Defaults to
            ``(4, 32, 32)``.
        d_model (int): Dimension of the underlying transformer decoder.
            See :py:class:`torchmultimodal.models.gpt.TransformerDecoderLayer`. Defaults to ``768``.
        n_head (int): Number of attention heads used by the transformer decoder. Defaults to ``8``.
        dropout (float): Dropout probability used by the projection layer of the transformer decoder.
            Defaults to ``0.2``.
        attn_dropout (float): Dropout probability used by the attention layer of the transformer decoder.
            Defaults to ``0.3``.
        num_decoder_layers (int): Number of transformer decoder layers. Defaults to ``12``.
        use_gpt_init (bool): Whether uses parameter initialization of GPT model. Defaults to ``True``.
        text_pretrained_tokenizer_url (str): Remote location of the pretrained text tokenizer file. Defaults
            to `"MUGEN pretrained tokenizer file
            "<https://pytorch.s3.amazonaws.com/models/multimodal/mugen/tokenizer-coinrun_1024.json>`_.
        video_vqvae_pretrained_model_key (str, optional): Key to select the pretrained MUGEN VideoVQVAE weights
            file. For allowed values, see :py:module:`examples/mugen/generation/video_vqvae.py`.

    Returns:
        An instance of :py:class:`torchmultimodal.models.gpt.MultimodalGPT`.
    """

    # builds text tokenizer from pre-trained
    text_tokenizer_local_path = _PATH_MANAGER.get_local_path(
        text_pretrained_tokenizer_url
    )
    tokenizer = Tokenizer.from_file(text_tokenizer_local_path)

    # builds text tokenizer
    text_tokenizer = TextTokenizer(
        context_len=text_seq_len,
        d_model=d_model,
        tokenizer=tokenizer,
    )
    num_text_tokens = text_tokenizer.num_text_tokens

    # builds video tokenizer
    video_vqvae = video_vqvae_mugen(
        pretrained_model_key=video_vqvae_pretrained_model_key,
        freeze_model=True,
    )
    video_vqvae.eval()
    num_video_tokens = video_vqvae.num_embeddings  # size of the codebook

    # derives the expected latent shape from video input shape
    video_input_shape = (video_seq_len, resolution, resolution)
    video_latent_shape = latent_shape(video_input_shape, downsample)
    video_vqvae_latent_shape = video_vqvae.latent_shape(video_input_shape)
    # video vqvae will apply convolutions to the input shape which effectively
    # reduces the size by ``dim//stride`` after each layer
    # sanity check that the expected and actual latent shapes are consistent
    if video_latent_shape != video_vqvae_latent_shape:
        raise ValueError(
            f"Latent shape derived from video inputs: {video_latent_shape} "
            f"does not match that of video vqvae: {video_vqvae_latent_shape}"
        )

    # builds text embedding projection: text_emb is already of output shape `d_model`
    # generally a projection layer is needed to bridge the tokenizer and
    # `torchmultimodal.models.gpt.MultimodalTransformerDecoder`, see `video_projection`
    text_projection = nn.Identity()

    # builds video embedding projection
    video_projection = nn.Linear(video_vqvae.embedding_dim, d_model, bias=False)

    # builds multimodal decoder
    text_pos_emb = nn.Embedding(text_seq_len, d_model)
    video_pos_emb = BroadcastedPositionEmbedding(video_latent_shape, d_model)
    attention_layer = SelfAttention(attn_dropout=attn_dropout)
    decoder_layer = TransformerDecoderLayer(
        d_model, n_head, dropout, attn_module=attention_layer
    )
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    right_shift = RightShift(d_model)
    mm_decoder = MultimodalTransformerDecoder(
        text_pos_emb, video_pos_emb, decoder, right_shift
    )

    return MultimodalGPT(
        d_model=d_model,
        num_in_tokens=num_text_tokens,
        num_out_tokens=num_video_tokens,
        latent_shape=video_latent_shape,
        in_tokenizer=text_tokenizer,
        out_tokenizer=video_vqvae,
        mm_decoder=mm_decoder,
        in_projection=text_projection,
        out_projection=video_projection,
        use_gpt_init=use_gpt_init,
    )


def latent_shape(
    input_shape: Tuple[int, ...], downsample: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Derives latent shape of video inputs after VQ-VAE encoding"""
    return tuple([s // d for s, d in zip(input_shape, downsample)])


class TextTokenizer(nn.Module):
    """Converts between text and tokens / embedings

    Wrapper around the tokenizer to be consistent with the API required by
    :py:class:`torchmultimodal.models.gpt.MultimodalGPT`. It also contains the
    embedding layer to enable lookup by token ids.
    """

    def __init__(
        self,
        context_len: int,
        d_model: int,
        tokenizer: nn.Module,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.encode("[PAD]").ids[0]  # type: ignore
        self.vocab_size = self.tokenizer.get_vocab_size()  # type: ignore
        self.context_len = context_len
        # MUGEN treats padding as unique ids so adding them to the total text tokens
        # https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/gpt/gpt.py#L44
        self.num_text_tokens = self.vocab_size + context_len
        self.embedding = nn.Embedding(self.num_text_tokens, d_model)

    def text_to_tokens(self, sentences: List[str]) -> Tensor:
        """Pads the sentences to be of equal lengths"""
        tokens = [
            self.tokenizer.encode(sentence.strip().lower() + " [SEP]")  # type: ignore
            for sentence in sentences
        ]
        token_ids = [t.ids[: self.context_len] for t in tokens]
        # pad each sentence to be of length `context_len`
        for i, t in enumerate(token_ids):
            t += [self.pad_id] * (self.context_len - len(t))
            token_ids[i] = t

        return torch.Tensor(token_ids).type(torch.int64)

    def encode(self, sentences: List[str], device: str) -> Tensor:
        """Encodes sentences to token ids"""
        token_ids = self.text_to_tokens(sentences).to(device)
        # bump padding token ids by vocab_size so that they do not coincide with un-padded token ids
        # and that the padding token ids themselves are unique
        unique_pad_ids = torch.arange(self.context_len, device=device) + self.vocab_size
        token_ids = torch.where(token_ids == self.pad_id, unique_pad_ids, token_ids)
        return token_ids

    def _filter_token_ids(self, token_ids: List[int]) -> List[Optional[int]]:
        """Filters out token ids out side of vocab"""
        return [
            token_id
            for token_id in token_ids
            if token_id > 0 and token_id <= self.vocab_size
        ]

    def decode(self, token_ids: Tensor) -> List[str]:
        """Decodes token ids back to sentences"""
        sentences = []
        for _token_ids in token_ids:  # iterate over batches
            _token_ids = self._filter_token_ids(_token_ids.tolist())
            sentence = self.tokenizer.decode(_token_ids)  # type: ignore
            sentences.append(sentence)

        return sentences

    def lookup(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids)
