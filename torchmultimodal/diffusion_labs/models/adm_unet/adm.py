# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchmultimodal.diffusion_labs.models.adm_unet.attention_block import (
    adm_attn_block,
)
from torchmultimodal.diffusion_labs.models.adm_unet.res_block import (
    adm_res_block,
    adm_res_downsample_block,
    adm_res_upsample_block,
)
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm
from torchmultimodal.modules.layers.position_embedding import (
    SinusoidalPositionEmbeddings,
)


class ADMUNet(nn.Module):
    """Ablated Diffusion Model as described in "Diffusion Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233)

    ADM consists of three major components:
        1) the timestep encoder, which embeds the timestep provided as an int
        2) the conditional projections, which project any conditional inputs such as class embeddings, CLIP image/text embeddings
        3) the UNet, which processes the above and a noised input, x, to generate an unnoised image.

    The UNet composes all the blocks for the downsampling encoder, bottleneck, and upsampling encoder in the ADM UNet.
    Constructs the network by adding residual blocks, attention blocks, and up/downsampling blocks for every layer
    based on user specifications.

    The UNet follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233).

    UNet code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L396

    Structure for the encoder and decoder:
    x -> ((ResBlock + MHA) * num_res -> ResDown/UpBlock) * num_layers -> h

    Overall UNet structure:
    x -> encoder -> bottleneck -> decoder -> out

    Code ref:
        https://github.com/lucidrains/DALLE2-pytorch/blob/c6c3882dc165914413ca97176b3a0103af1d7048/dalle2_pytorch/dalle2_pytorch.py#L1856

    Attributes:
        channels_per_layer (List[int]): list of channels for each layer. Total number of layers is determined by
            length of this list. Each item must be divisible by number of groups used in GroupNorm layers. ADMResBlock
            and ADMAttentionBlock use 32 as a default.
        num_resize (int): number of layers that will include downsampling or upsampling
        num_res_per_layer (int): number of residual blocks per layer
        use_attention_for_layer (List[bool]): list indicating whether to include attention after res block for each layer.
            Must match the length of channels_per_layer.
        dim_res_cond (int): dimensionality of conditional projection layer in each res block
        dim_attn_cond (Optional[int]): dimensionality of cross attention inputs used in attention blocks. If None, do not
            use conditional inputs in attention blocks
        embed_dim (Optional[int]): input dimension of conditional embeddings. Can be used to
            construct default res_cond_proj and attn_cond_proj if they are not passed. Defaults to None
        in_channels (int): number of channels in input image. Defaults to 3.
        out_channels (int): number of channels in output image. Defaults to 3.
        time_embed_dim (Optional[int]): dimensionality of timestep embedding. Used to construct
            default timestep encoder if timestep_encoder is None. Defaults to None
        timestep_encoder (Optional[nn.Module]): module that will embed an integer timestep. Expected output shape is [b, c]
            where c is the embedding dim. Defaults to None
        res_cond_proj (Optional[nn.ModuleDict]): optional dict that maps user-defined keys to modules to project
            conditional_inputs before passed into the UNet. Keys should align with the dict passed into conditional_inputs.
            Output of projections are SUMMED with the embedded timestep and passed into the residual blocks of the unet,
            if using ADMUNet. Embedding dims of outputs need to match embedding dim of timestep_encoder output.
        attn_cond_proj (Optional[nn.ModuleDict]): optional dict that maps user-defined keys to modules to project
            conditional_inputs before passed into the UNet. Keys should align with the dict passed into conditional_inputs.
            Output of projections are CONCATENATED in the sequence dimension and passed into the attention blocks of the unet,
            if using ADMUNet.
        predict_variance_value (bool): If True, expects unet model to output doubled channels, second half of which are the
            predicted variances for q, the posterior of the forward diffusion process. This corresponds to the v variable
            in Eq. 15 of "Improved Denoising Diffusion Probabilistic Models". If False, simply return the unet output.
            Default is True.
        variance_value_transform (Optional[Callable]): optional function to transform the model output towards an intended
            output range. The variance_value is intended to be a value from 0 to 1. If the model is trained to output values
            from -n to n, then a recommended transform would be lambda x: (x + n) / 2n.


    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        timestep (Tensor): diffusion timesteps of shape [b, ]
        conditional_inputs (Optional[Dict[str, Tensor]]): optional Tensor inputs to condition the unet model on. Keyed by
            user-defined strings that should align with res_cond_proj or attn_cond_proj, otherwise the value is unused.
            Expected shape of tensors are [b, c], where c is the embedding dim of the Tensor.
    """

    def __init__(
        self,
        *,
        channels_per_layer: List[int],
        num_resize: int,
        num_res_per_layer: int,
        use_attention_for_layer: List[bool],
        dim_res_cond: int,
        dim_attn_cond: Optional[int] = None,
        embed_dim: Optional[int] = None,
        embed_name: str = "context",
        in_channels: int = 3,
        out_channels: int = 3,
        time_embed_dim: Optional[int] = None,
        timestep_encoder: Optional[nn.Module] = None,
        res_cond_proj: Optional[nn.ModuleDict] = None,
        attn_cond_proj: Optional[nn.ModuleDict] = None,
        predict_variance_value: bool = True,
        variance_value_transform: Optional[Callable] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        if timestep_encoder is None:
            assert (
                time_embed_dim is not None
            ), "Must pass either timestep_encoder or time_embed_dim"
            self.timestep_encoder = self._create_timestep_encoder(
                time_embed_dim, dim_res_cond
            )
        else:
            self.timestep_encoder = timestep_encoder
        self.embed_name = embed_name
        if res_cond_proj is None and embed_dim is not None and dim_res_cond is not None:
            self.res_cond_proj = self._create_res_cond_proj(embed_dim, dim_res_cond)
        else:
            self.res_cond_proj = res_cond_proj
        if (
            attn_cond_proj is None
            and embed_dim is not None
            and dim_attn_cond is not None
        ):
            self.attn_cond_proj = self._create_attn_cond_proj(embed_dim, dim_attn_cond)
        else:
            self.attn_cond_proj = attn_cond_proj

        self.predict_variance_value = predict_variance_value
        self.variance_value_transform = variance_value_transform or nn.Identity()

        if len(channels_per_layer) != len(use_attention_for_layer):
            raise ValueError(
                f"Attention or number of channels not specified for each layer, expected {len(channels_per_layer)} layers"
            )
        if len(channels_per_layer) < num_resize:
            raise ValueError(
                f"Not enough channels specified, cannot have less than num_resize ({num_resize})"
            )

        self.channels_per_layer = channels_per_layer
        self.num_resize = num_resize
        self.num_res_per_layer = num_res_per_layer
        self.use_attention_for_layer = use_attention_for_layer
        self.dim_res_cond = dim_res_cond
        self.dim_attn_cond = dim_attn_cond
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down, down_channels = self._create_downsampling_encoder()
        self.bottleneck = self._create_bottleneck(down_channels[-1])
        self.up = self._create_upsampling_decoder(down_channels)

    def _create_timestep_encoder(
        self, time_embed_dim: int, cond_embed_dim: int
    ) -> nn.Module:
        return nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim=time_embed_dim),
            nn.Linear(time_embed_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

    def _create_res_cond_proj(
        self, embed_dim: int, cond_embed_dim: int
    ) -> nn.ModuleDict:
        return nn.ModuleDict({self.embed_name: nn.Linear(embed_dim, cond_embed_dim)})

    def _create_attn_cond_proj(
        self, embed_dim: int, cond_embed_dim: int
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                self.embed_name: nn.Sequential(
                    nn.Linear(
                        embed_dim, cond_embed_dim * 4
                    ),  # four tokens of context as per paper ref
                    nn.Unflatten(-1, (4, cond_embed_dim)),
                )
            }
        )

    def _create_downsampling_encoder(self) -> Tuple[nn.ModuleList, List]:
        # Keep track of output channels of every block for thru connections to decoder
        down_channels = []
        # Use ADMStack for conv layer so we can pass in conditional inputs and ignore them
        init_conv = ADMStack()
        init_conv.append_simple_block(
            nn.Conv2d(
                self.in_channels,
                self.channels_per_layer[0],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        down_channels.append(self.channels_per_layer[0])

        stacks = []
        layer_in_channels = self.channels_per_layer[0]
        # you can pass more channels than downsample layers, in which case
        # the last few layers will not use downsampling blocks. so we
        # should track the number of layers from channel list
        for layer_num, layer_channels in enumerate(self.channels_per_layer):
            layer_out_channels = layer_channels
            # First create res blocks for the layer, with attention in between if specified
            for block_num in range(self.num_res_per_layer):
                if self.use_attention_for_layer[layer_num]:
                    stacks.append(
                        adm_stack_res_attn(
                            in_channels=layer_in_channels,
                            out_channels=layer_out_channels,
                            dim_res_cond=self.dim_res_cond,
                            dim_attn_cond=self.dim_attn_cond,
                        )
                    )
                else:
                    stacks.append(
                        adm_stack_res(
                            in_channels=layer_in_channels,
                            out_channels=layer_out_channels,
                            dim_cond=self.dim_res_cond,
                        )
                    )
                down_channels.append(layer_out_channels)
                layer_in_channels = layer_out_channels
            # Now create the down/upsampling res block
            if layer_num < self.num_resize:
                stacks.append(
                    adm_stack_res_down(
                        num_channels=layer_out_channels, dim_cond=self.dim_res_cond
                    )
                )
                down_channels.append(layer_out_channels)

        net = nn.ModuleList([init_conv] + stacks)
        return net, down_channels

    def _create_bottleneck(self, num_channels: int) -> nn.Module:
        in_resblock = adm_res_block(
            in_channels=num_channels,
            out_channels=num_channels,
            dim_cond=self.dim_res_cond,
        )
        mid_attention = adm_attn_block(
            num_channels=num_channels, dim_cond=self.dim_attn_cond
        )
        out_resblock = adm_res_block(
            in_channels=num_channels,
            out_channels=num_channels,
            dim_cond=self.dim_res_cond,
        )
        adm_stack = ADMStack()
        adm_stack.append_residual_block(in_resblock)
        adm_stack.append_attention_block(mid_attention)
        adm_stack.append_residual_block(out_resblock)
        return adm_stack

    def _create_upsampling_decoder(self, down_channels: List[int]) -> nn.ModuleList:
        # reverse so it's easier to iterate when going up the decoder
        up_channels_per_layer = list(reversed(self.channels_per_layer))
        up_attention_for_layer = list(reversed(self.use_attention_for_layer))

        stacks = []
        layer_in_channels = up_channels_per_layer[0]
        # you can pass more channels than downsample layers, in which case
        # the last few layers will not use downsampling blocks. so we
        # should track the number of layers from channel list
        for layer_num, layer_channels in enumerate(up_channels_per_layer):
            layer_out_channels = layer_channels
            # Code ref uses + 1 res blocks in upsampling decoder to add extra res block before upsampling
            for block_num in range(self.num_res_per_layer + 1):
                thru_channels = down_channels.pop() if down_channels else 0

                if up_attention_for_layer[layer_num]:
                    stacks.append(
                        adm_stack_res_attn(
                            in_channels=layer_in_channels + thru_channels,
                            out_channels=layer_out_channels,
                            dim_res_cond=self.dim_res_cond,
                            dim_attn_cond=self.dim_attn_cond,
                        )
                    )
                else:
                    stacks.append(
                        adm_stack_res(
                            in_channels=layer_in_channels + thru_channels,
                            out_channels=layer_out_channels,
                            dim_cond=self.dim_res_cond,
                        )
                    )
                layer_in_channels = layer_out_channels
            # Now create the down/upsampling res block
            if layer_num < self.num_resize:
                stacks[-1].append_residual_block(
                    adm_res_upsample_block(
                        num_channels=layer_out_channels,
                        dim_cond=self.dim_res_cond,
                    )
                )

        out_conv = ADMStack()
        out_conv.append_simple_block(
            nn.Sequential(
                Fp32GroupNorm(32, up_channels_per_layer[-1]),
                nn.SiLU(),
                nn.Conv2d(
                    up_channels_per_layer[-1],
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

        net = nn.ModuleList(stacks + [out_conv])
        return net

    def _get_conditional_projections(
        self,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        res_cond = []
        attn_cond = []
        # Prevent for loop from running if conditional_inputs is None
        conditional_inputs = (
            conditional_inputs if conditional_inputs is not None else {}
        )

        t_embed = self.timestep_encoder(timestep)
        t_embed_dim = t_embed.shape[-1]
        res_cond.append(t_embed)

        for key in conditional_inputs:
            additional_input = conditional_inputs[key]

            if self.res_cond_proj is not None and key in self.res_cond_proj:
                cond_proj = self.res_cond_proj[key](additional_input)
                if cond_proj.shape[-1] != t_embed_dim:
                    raise ValueError(
                        f"Embedding dim of res_cond_proj for {key} incompatible with timestep_encoder: "
                        f"expected {t_embed_dim}, got {cond_proj.shape[-1]}"
                    )
                res_cond.append(cond_proj)

            if self.attn_cond_proj is not None and key in self.attn_cond_proj:
                cond_proj = self.attn_cond_proj[key](additional_input)
                # Append as [b, c] -> [b, t, c]
                if len(cond_proj.shape) < 3:
                    cond_proj = cond_proj.unsqueeze(1)
                attn_cond.append(cond_proj)

        res_cond = torch.stack(res_cond).sum(dim=0)
        # Concat on sequence dimension
        attn_cond = torch.concat(attn_cond, dim=1) if attn_cond else None
        return res_cond, attn_cond

    def _get_variance_value(
        self, x: Tensor, h: Tensor
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # If model is predicting variance, then it should be configured to output double the channels as input
        if self.predict_variance_value:
            if h.shape[1] != x.shape[1] * 2:
                raise ValueError(
                    f"unet is not configured to predict variance values. "
                    f"Expected output channel dim to be {x.shape[1] * 2}, got {h.shape[1]}"
                )
            # Split in half in channel dim
            prediction, variance_value = torch.chunk(h, 2, dim=1)
            variance_value = self.variance_value_transform(variance_value)
        else:
            prediction = h
            variance_value = None
        return prediction, variance_value

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ) -> DiffusionOutput:
        (
            res_conditional_embedding,
            attn_conditional_embedding,
        ) = self._get_conditional_projections(timestep, conditional_inputs)

        hidden_states = []
        h = x
        for block in self.down:
            h = block(h, res_conditional_embedding, attn_conditional_embedding)
            hidden_states.append(h)
        h = self.bottleneck(h, res_conditional_embedding, attn_conditional_embedding)
        for block in self.up:
            if hidden_states:
                h = torch.cat([h, hidden_states.pop()], dim=1)
            h = block(h, res_conditional_embedding, attn_conditional_embedding)

        prediction, variance_value = self._get_variance_value(x, h)

        return DiffusionOutput(
            prediction=prediction,
            variance_value=variance_value,
        )


class ADMStackModuleType(Enum):
    ResidualBlock = 0
    AttentionBlock = 1
    SimpleBlock = 2


class ADMStack(nn.Module):
    """A container that acts like a ModuleList of ADM blocks and handles passing timestep and
    context embeddings correctly to its children. Usually, blocks such as residual blocks consume
    timestep embeddings, while attention blocks consume optional contextual embeddings in addition
    to the input x. This container allows us to wrap the modules so that they can be stacked in a
    `nn.Sequential`, in order to simplify the code for the `forward` method.

    We have to implement the stack in this way rather than inherting from `nn.ModuleList` to
    avoid FSDP/Activation Checkpointing/PT2 incompatibility issues.

    Code ref: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L35
    """

    def __init__(self) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")
        self._module_list = nn.ModuleList()
        self._module_types: List[ADMStackModuleType] = []

    def append_attention_block(self, module: nn.Module) -> None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.AttentionBlock)

    def append_residual_block(self, module: nn.Module) -> None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.ResidualBlock)

    def append_simple_block(self, module: nn.Module) -> None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.SimpleBlock)

    def forward(
        self,
        x: Tensor,
        residual_conditional_embedding: Tensor,
        attention_conditional_embedding: Optional[Union[Tensor, Sequence[Tensor]]],
    ) -> Tensor:
        h = x
        for name, block in zip(self._module_types, self._module_list):  # noqa: B905
            if name == ADMStackModuleType.ResidualBlock:
                h = block(h, residual_conditional_embedding)
            elif name == ADMStackModuleType.AttentionBlock:
                h = block(h, attention_conditional_embedding)
            else:
                h = block(h)
        return h


def adm_stack_res(in_channels: int, out_channels: int, dim_cond: int) -> nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(
        adm_res_block(
            in_channels=in_channels,
            out_channels=out_channels,
            dim_cond=dim_cond,
        )
    )
    return adm_stack


def adm_stack_res_attn(
    in_channels: int,
    out_channels: int,
    dim_res_cond: int,
    dim_attn_cond: Optional[int] = None,
) -> nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(
        adm_res_block(
            in_channels=in_channels,
            out_channels=out_channels,
            dim_cond=dim_res_cond,
        )
    )
    adm_stack.append_attention_block(
        adm_attn_block(
            num_channels=out_channels,
            dim_cond=dim_attn_cond,
        )
    )
    return adm_stack


def adm_stack_res_down(num_channels: int, dim_cond: int) -> nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(
        adm_res_downsample_block(
            num_channels=num_channels,
            dim_cond=dim_cond,
        )
    )
    return adm_stack


def adm_unet(
    *,
    # ADM args
    time_embed_dim: int = 512,
    cond_embed_dim: int = 2048,
    embed_dim: int = 768,
    embed_name: str = "context",
    predict_variance_value: bool = True,
    # ADMUNet args
    image_channels: int = 4,
    depth: int = 512,
    num_resize: int = 3,
    num_res_per_layer: int = 3,
) -> nn.Module:
    """Constructs a conditional ADM U-Net

    Consists of an ADM UNet diffusion model conditioned on CLIP image embeddings.

    Follows parameters from paper: https://arxiv.org/abs/2204.06125


    Args:
        time_embed_dim (int): desired dimensionality of timestep embedding
        cond_embed_dim (int): desired dimensionality of conditional input embeddings
        embed_dim (int): expected dimensionality of conditional image embeddings
        embed_name (str): name of conditional image embeddings
        predict_variance_value (bool): if True, will double UNet's output channel dim to predict variance values of
            diffusion process
        image_channels (int): channel dim of input images
        depth (int): channel dim of UNet convolutional modules. Expands everytime resolution is downscaled.
            Must be divisible by number of groups used in GroupNorm layers. ADMResBlock
            and ADMAttentionBlock use 32 as a default.
        num_resize (int): number of times resolution will be scaled
        num_res_per_layer (int): number of residual blocks per resolution
    """
    # Construct UNet
    in_channels = image_channels

    # If predicting variance, double the channel dim of UNet output and use those values as variance
    out_channels = image_channels * 2 if predict_variance_value else image_channels

    # e.g., if depth is 512 and num_resize is 3, then channels are [512, 1024, 1536, 2048]
    channels_per_layer = [depth * (i + 1) for i in range(num_resize + 1)]
    # Assuming image size is 64x64, use attention at resolutions 32x32, 16x16, 8x8
    use_attention_per_layer = [False] + [True] * num_resize

    return ADMUNet(
        channels_per_layer=channels_per_layer,
        num_resize=num_resize,
        num_res_per_layer=num_res_per_layer,
        use_attention_for_layer=use_attention_per_layer,
        dim_res_cond=cond_embed_dim,
        dim_attn_cond=cond_embed_dim,
        embed_dim=embed_dim,
        embed_name=embed_name,
        in_channels=in_channels,
        out_channels=out_channels,
        time_embed_dim=time_embed_dim,
        predict_variance_value=predict_variance_value,
        variance_value_transform=lambda x: (x + 1) / 2,
    )
