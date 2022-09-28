# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on https://github.com/openai/CLIP/blob/main/clip/model.py

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.modules.layers.activation import SiLU
from torchmultimodal.modules.layers.normalizations import Fp32LayerNorm


EXPANSION = 4


class CLIPViTEncoder(nn.Module):
    """
    Vision transformer encoder for CLIP.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        patch_size (int): The dimension of each patch
        image_size(int): The size (width==height) of input image
        width (int): Dimensionality of the encoder layers and the pooler layer
        heads (int): Number of attention heads for each attention layer in the Transformer encoder
        layers (int): Number of hidden layers in the Transformer encoder

    Inputs:
        x (Tensor): image tensor with dimensions B x C(3) x image_size x image_size
    """

    def __init__(
        self,
        embedding_dim: int,
        patch_size: int,
        image_size: int,
        width: int,
        heads: int,
        layers: int,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.image_size = image_size

        scale = width**-0.5
        self.cls_token_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((image_size // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = Fp32LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dropout=0.0,
            activation=SiLU(),
            norm_first=True,
            dim_feedforward=4 * width,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )

        self.ln_post = Fp32LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:

        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(
                f"Expected input with width and height as {self.image_size}, found {x.size(2)} by {x.size(3)} "
            )
        if x.size(1) != 3:
            raise ValueError(f"Expected 3 channels found {x.size(1)}")

        # B x C x image_size x image_size => B x width (out_channel) x patch_size x patch_size
        x = self.conv(x)

        # B x width x patch_size x patch_size => B x width x patch_size ** 2
        x = torch.flatten(x, start_dim=2)

        # B x width x patch_size ** 2 => B x patch_size ** 2 x width
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [
                self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = self.encoder(x)

        # Take embedding of the cls token
        x = self.ln_post(x[:, 0, :])
        x = x @ self.projection
        return x


class ResNetForCLIPBottleneck(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        # all conv layers have stride 1.
        # an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * EXPANSION, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * EXPANSION)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * EXPANSION:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * EXPANSION,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * EXPANSION)),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ResNetForCLIP(nn.Module):
    """Modified ResNet used by CLIP.

    Based on https://github.com/openai/CLIP/blob/main/clip/model.py#L93, this class
    differs from Torchvision's ResNet in the following ways:
    - There are now 3 "stem" convolutions as opposed to 1, with an
        average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
        prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.

    Args:
        layers (Tuple[int]):
        output_dim (int): dimension of output tensor
        heads (int): number of heads in the attention pooling layer
        input_resolution (int): resolution of image input to encoder
        width (int): ResNet width
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs:
        x (Tensor): Tensor containing image features
    """

    def __init__(
        self,
        layers: Tuple[int, int, int, int] = (3, 4, 6, 3),
        output_dim: int = 512,
        heads: int = 1024,
        input_resolution: int = 224,
        width: int = 64,
        use_clip_init: bool = True,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

        if use_clip_init:
            self.initialize_parameters()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        layers = [ResNetForCLIPBottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * EXPANSION
        for _ in range(1, blocks):
            layers.append(ResNetForCLIPBottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_parameters(self) -> None:
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
        # Zero-initialize each block's third batch normalization weights
        # Based on CLIP initialization in https://git.io/JDbGX
        for resnet_block in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def forward(self, x: Tensor) -> Tensor:
        def stem(x: Tensor) -> Tensor:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x
