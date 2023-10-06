# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, NamedTuple, Optional, Tuple, Union

import torch

from torch import nn, Tensor
from torchmultimodal.models.masked_auto_encoder.position_embeddings import (
    get_2d_sin_cos_embeddings,
)
from torchmultimodal.models.masked_auto_encoder.swin_decoder import SwinTransformer
from torchmultimodal.modules.encoders.vision_transformer import (
    VisionTransformer,
    vit_b_16,
    vit_l_16,
)
from torchmultimodal.modules.layers.patch_embedding import PatchEmbeddings
from torchmultimodal.modules.layers.transformer import (
    TransformerEncoder,
    TransformerOutput,
)


MAE_MODEL_MAPPING = {
    "vit_b16_image": "https://download.pytorch.org/models/multimodal/mae/mae_pretrained_vit_base.pth",
    "vit_l16_image": "https://download.pytorch.org/models/multimodal/mae/mae_pretrained_vit_large.pth",
    "vit_b16_audio": "https://download.pytorch.org/models/multimodal/audio_mae/audio_mae_pretrained_vit_base.pth",
}


class MAEOutput(NamedTuple):
    encoder_output: Union[TransformerOutput, Tensor]
    decoder_pred: Optional[Tensor] = None
    label_patches: Optional[Tensor] = None
    mask: Optional[Tensor] = None


class MaskedAutoEncoder(nn.Module):
    """
    MAE (https://arxiv.org/abs/2111.06377) is a pretraining technique to mask out patches of the input
    before passing through the encoder and then using a decoder to predict the masked patches
    The code has been adapted from the original implementation https://github.com/facebookresearch/mae

    Args:
        encoder_transformer (nn.Module): instance of encoder transformer
        decoder_transformer (nn.Module): instance of decoder transformer
        input_size (Union[int, Tuple[int,int]): size of the input. if tuple, the format should be height,width.
        If an int, a square input is assumed. Default: 224
        patch_size (int): size of the patches. Default: 16
        num_channels (int): number of input channels. Default: 3
        embed_dim (int): embedding dim of input to the encoder transformer (or output dim of patch embedding). Default: 768
        masking_ratio (float): ratio of patches to mask. Default: 0.75
        decoder_embed_dim (int): embedding dim of the input to the decoder transformer. Default: 512
    """

    def __init__(
        self,
        encoder_transformer: nn.Module,
        decoder_transformer: nn.Module,
        input_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
        masking_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        use_cls_in_decoder: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embeddings = PatchEmbeddings(
            image_size=input_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=embed_dim,
            patch_drop_rate=masking_ratio,
        )
        self.embeddings.position_embeddings.requires_grad = False

        self.encoder = encoder_transformer

        self.decoder_embed = DecoderEmbeddings(
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            image_size=input_size,
            patch_size=patch_size,
        )
        self.decoder_embed.position_embeddings.requires_grad = False

        self.decoder_transformer = decoder_transformer
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * num_channels)
        self.use_cls_in_decoder = use_cls_in_decoder
        self._initialize_weights(input_size, embed_dim, decoder_embed_dim)

    def _initialize_weights(
        self,
        input_size: Union[int, Tuple[int, int]],
        encoder_embed_dim: int,
        decoder_embed_dim: int,
    ) -> None:
        if isinstance(input_size, int):
            input_h = input_w = input_size
        else:
            input_h, input_w = input_size
        num_patches_h = input_h // self.patch_size
        num_patches_w = input_w // self.patch_size
        self.embeddings.position_embeddings.data = get_2d_sin_cos_embeddings(
            encoder_embed_dim, (num_patches_w, num_patches_h)
        )
        self.decoder_embed.position_embeddings.data = get_2d_sin_cos_embeddings(
            decoder_embed_dim, (num_patches_w, num_patches_h)
        )

        # initialize embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.embeddings.conv_projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.embeddings.cls_token, std=0.02)
        torch.nn.init.normal_(self.decoder_embed.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _patchify_input(self, x: Tensor) -> Tensor:
        # patchify the input tensor with the output shape = bsz x num_patch x (patch_area * channels)
        bsz, channels, height, width = x.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        label_patches = x.reshape(
            bsz,
            channels,
            num_patches_h,
            self.patch_size,
            num_patches_w,
            self.patch_size,
        )
        label_patches = torch.einsum("nchpwq->nhwpqc", label_patches)
        label_patches = label_patches.reshape(
            bsz, num_patches_h * num_patches_w, self.patch_size**2 * channels
        )
        return label_patches

    def forward(self, x: Tensor) -> MAEOutput:
        """
        Args:
            x (Tensor): input tensor with shape bsz x channels x height x width
        Returns:
            output of MAEOutput type where encoder_output gives the output from the encoder,
            decoder_pred gives prediction from the decoder followed by linear head,
            mask indicates the masked out patches i.e. 1 refers to masked patches and 0 refers to unmasked patches
            label_patches indicates the patchified ground truth pixels

        """
        embedding_out = self.embeddings(x)
        encoder_out = self.encoder(embedding_out.embeddings)
        if not self.training:
            # TODO: check if error should be raised is masking ratio != 0 here
            return MAEOutput(encoder_out)
        decoder_embedding = self.decoder_embed(
            encoder_out.last_hidden_state, embedding_out.ids_restore
        )

        decoder_input = decoder_embedding
        if not self.use_cls_in_decoder:
            decoder_input = decoder_input[:, 1:, :]

        decoder_out = self.decoder_transformer(decoder_input)
        pred = self.decoder_pred(decoder_out.last_hidden_state)

        if self.use_cls_in_decoder:
            pred = pred[:, 1:, :]

        label_patches = self._patchify_input(x)

        return MAEOutput(
            encoder_output=encoder_out,
            decoder_pred=pred,
            label_patches=label_patches,
            mask=embedding_out.random_mask,
        )


class DecoderEmbeddings(nn.Module):
    """
    Construct the decoder embeddings from encoder embeddings.
    Args:
        encoder_embed_dim (int): Input dim for decoder embedding i.e. output dim of the encoder.
        decoder_embed_dim (int): output dim for decoder embedding.
        image_size (Union[int, Tuple[int, int]]): Size of the original input image. If set to an int, we assume a square input.
         Defaults to 224.
        patch_size (int): Patch size for the decoder.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        decoder_embed_dim: int,
        image_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )

    def forward(
        self,
        x: Tensor,
        ids_restore: Tensor,
    ) -> Tensor:
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # adding positional embeddings
        x = x + self.position_embeddings

        return x


def image_mae(
    *,
    # patch embedding
    image_size: int = 224,
    patch_size: int = 16,
    masking_ratio: float = 0.75,
    # encoder
    encoder_layers: int = 12,
    encoder_hidden_dim: int = 768,
    encoder_heads: int = 12,
    encoder_dim_feedforward: int = 3072,
    encoder_layer_norm_eps: float = 1e-6,
    encoder_activation: Callable = nn.GELU,
    encoder_final_layer_norm_eps: float = 1e-6,
    # decoder
    decoder_layers: int = 8,
    decoder_hidden_dim: int = 512,
    decoder_heads: int = 16,
    decoder_dim_feedforward: int = 2048,
    decoder_layer_norm_eps: float = 1e-6,
    decoder_activation: Callable = nn.GELU,
    decoder_final_layer_norm_eps: float = 1e-6,
) -> MaskedAutoEncoder:
    """
    Helper function to build the image mae model instantiation as described in the paper
    with the encoder and decoder transformer similar to vision transformer.
    Args:
        image_size (int): size of the input. Default: 224
        patch_size (int): size of the patches. Default: 16
        masking_ratio (float): ratio of patches to mask. Default: 0.75
        encoder_layers(int): number of encoder layers. Default: 12
        encoder_hidden_dim (int): hidden dim of the encoder transformer. Default: 768
        encoder_heads (int): number of encoder heads. Default: 12
        encoder_dim_feedforward (int): hidden dim of the encoder transformer feedforward layer. Default: 3072
        encoder_activation (Callable): activation function for encoder layers. Default: nn.GELU
        encoder_layer_norm_eps (float): epsilon for encoder layer normalization. Default: 1e-6
        encoder_final_layer_norm_eps (float): epsilon for encoder final layer normalization. Default: 1e-6
        decoder_layers(int): number of decoder layers. Default: 8
        decoder_hidden_dim (int): hidden dim of the decoder transformer. Default: 512
        decoder_heads (int): number of decoder heads. Default: 16
        decoder_dim_feedforward (int): hidden dim of the decoder transformer feedforward layer. Default: 2048
        decoder_layer_norm_eps (float): epsilon for decoder layer normalization. Default: 1e-6
        decoder_activation (float): activation function for decoder layers. Default: nn.GELU
        decoder_final_layer_norm_eps (float): epsilon for decoder final layer normalization. Default: 1e-6

    """
    encoder_transformer = TransformerEncoder(
        n_layer=encoder_layers,
        d_model=encoder_hidden_dim,
        n_head=encoder_heads,
        dim_feedforward=encoder_dim_feedforward,
        final_layer_norm_eps=encoder_final_layer_norm_eps,
        layer_norm_eps=encoder_layer_norm_eps,
        norm_first=True,
        activation=encoder_activation,
    )
    decoder_transformer = TransformerEncoder(
        n_layer=decoder_layers,
        d_model=decoder_hidden_dim,
        n_head=decoder_heads,
        dim_feedforward=decoder_dim_feedforward,
        layer_norm_eps=decoder_layer_norm_eps,
        final_layer_norm_eps=decoder_final_layer_norm_eps,
        norm_first=True,
        activation=decoder_activation,
    )
    return MaskedAutoEncoder(
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
        input_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        embed_dim=encoder_hidden_dim,
        masking_ratio=masking_ratio,
        decoder_embed_dim=decoder_hidden_dim,
    )


def vit_l_16_image_mae() -> MaskedAutoEncoder:
    return image_mae(
        image_size=224,
        patch_size=16,
        masking_ratio=0.75,
        encoder_layers=24,
        encoder_hidden_dim=1024,
        encoder_heads=16,
        encoder_dim_feedforward=4096,
        decoder_layers=8,
        decoder_hidden_dim=512,
        decoder_heads=16,
        decoder_dim_feedforward=2048,
    )


def vit_b_16_image_mae_encoder(pretrained: bool = False) -> VisionTransformer:
    ckpt_path = MAE_MODEL_MAPPING["vit_b16_image"] if pretrained else None
    return vit_b_16(final_layer_norm_eps=None, ckpt_path=ckpt_path)


def vit_l_16_image_mae_encoder(pretrained: bool = False) -> VisionTransformer:
    ckpt_path = MAE_MODEL_MAPPING["vit_l16_image"] if pretrained else None
    return vit_l_16(final_layer_norm_eps=None, ckpt_path=ckpt_path)


def audio_mae(
    *,
    # patch embedding
    input_size: Tuple[int, int] = (1024, 128),
    patch_size: int = 16,
    masking_ratio: float = 0.8,
    # encoder
    encoder_layers: int = 12,
    encoder_hidden_dim: int = 768,
    encoder_heads: int = 16,
    encoder_dim_feedforward: int = 3072,
    encoder_layer_norm_eps: float = 1e-6,
    encoder_activation: Callable = nn.GELU,
    encoder_final_layer_norm_eps: float = 1e-6,
    # decoder
    window_size: Tuple[int, int] = (4, 4),
    decoder_layers: int = 16,
    decoder_hidden_dim: int = 512,
    decoder_heads: int = 16,
    decoder_dim_feedforward: int = 2048,
    decoder_layer_norm_eps: float = 1e-6,
    decoder_activation: Callable = nn.GELU,
    decoder_final_layer_norm_eps: float = 1e-6,
) -> MaskedAutoEncoder:
    """
    Helper function to build the standard audio mae model with the encoder similar to vision transformer\
    and decoder transformer similar to swin transformer.
    Args:
        image_size (Tuple[int, int]): (height, width) of the input. Default: (1024, 128)
        patch_size (int): size of the patches. Default: 16
        masking_ratio (float): ratio of patches to mask. Default: 0.8
        encoder_layers(int): number of encoder layers. Default: 12
        encoder_hidden_dim (int): hidden dim of the encoder transformer. Default: 768
        encoder_heads (int): number of encoder heads. Default: 16
        encoder_dim_feedforward (int): hidden dim of the encoder transformer feedforward layer. Default: 3072
        encoder_activation (Callable): activation function for encoder layers. Default: nn.GELU
        encoder_layer_norm_eps (float): epsilon for encoder layer normalization. Default: 1e-6
        encoder_final_layer_norm_eps (float): epsilon for encoder final layer normalization. Default: 1e-6
        decoder_layers(int): number of decoder layers. Default: 16
        decoder_hidden_dim (int): hidden dim of the decoder transformer. Default: 512
        decoder_heads (int): number of decoder heads. Default: 16
        decoder_dim_feedforward (int): hidden dim of the decoder transformer feedforward layer. Default: 2048
        decoder_layer_norm_eps (float): epsilon for decoder layer normalization. Default: 1e-6
        decoder_activation (float): activation function for decoder layers. Default: nn.GELU
        decoder_final_layer_norm_eps (float): epsilon for decoder final layer normalization. Default: 1e-6

    """
    encoder_transformer = TransformerEncoder(
        n_layer=encoder_layers,
        d_model=encoder_hidden_dim,
        n_head=encoder_heads,
        dim_feedforward=encoder_dim_feedforward,
        final_layer_norm_eps=encoder_final_layer_norm_eps,
        layer_norm_eps=encoder_layer_norm_eps,
        norm_first=True,
        activation=encoder_activation,
    )
    decoder_input_size = (input_size[0] // patch_size, input_size[1] // patch_size)
    decoder_transformer = SwinTransformer(
        n_layer=decoder_layers,
        input_dim=decoder_hidden_dim,
        feedforward_dim=decoder_dim_feedforward,
        num_heads=decoder_heads,
        input_size=decoder_input_size,
        window_size=window_size,
    )
    return MaskedAutoEncoder(
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
        input_size=input_size,
        patch_size=patch_size,
        num_channels=1,
        embed_dim=encoder_hidden_dim,
        masking_ratio=masking_ratio,
        decoder_embed_dim=decoder_hidden_dim,
        use_cls_in_decoder=False,
    )


def vit_s_16_audio_mae() -> MaskedAutoEncoder:
    return audio_mae(
        input_size=(1024, 128),
        patch_size=16,
        masking_ratio=0.8,
        encoder_layers=12,
        encoder_hidden_dim=384,
        encoder_heads=6,
        encoder_dim_feedforward=1536,
        decoder_layers=16,
        decoder_hidden_dim=512,
        decoder_heads=16,
        decoder_dim_feedforward=2048,
    )


def vit_b_16_audio_mae() -> MaskedAutoEncoder:
    return audio_mae(
        input_size=(1024, 128),
        patch_size=16,
        masking_ratio=0.8,
        encoder_layers=12,
        encoder_hidden_dim=768,
        encoder_heads=12,
        encoder_dim_feedforward=3072,
        decoder_layers=16,
        decoder_hidden_dim=512,
        decoder_heads=16,
        decoder_dim_feedforward=2048,
    )


def vit_l_16_audio_mae() -> MaskedAutoEncoder:
    return audio_mae(
        input_size=(1024, 128),
        patch_size=16,
        masking_ratio=0.8,
        encoder_layers=24,
        encoder_hidden_dim=1024,
        encoder_heads=16,
        encoder_dim_feedforward=4096,
        decoder_layers=16,
        decoder_hidden_dim=512,
        decoder_heads=16,
        decoder_dim_feedforward=2048,
    )


def vit_b_16_audio_mae_encoder(pretrained: bool = False) -> VisionTransformer:
    ckpt_path = MAE_MODEL_MAPPING["vit_b16_audio"] if pretrained else None
    return vit_b_16(
        final_layer_norm_eps=None,
        num_channels=1,
        image_size=(1024, 128),
        ckpt_path=ckpt_path,
    )
