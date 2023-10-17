# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchmultimodal.diffusion_labs.models.adm_unet.adm import ADMStack
from torchmultimodal.diffusion_labs.models.ldm.spatial_transformer import (
    SpatialTransformer,
)
from torchmultimodal.diffusion_labs.models.vae.res_block import adm_cond_proj, ResBlock
from torchmultimodal.diffusion_labs.models.vae.residual_sampling import (
    Downsample2D,
    Upsample2D,
)
from torchmultimodal.diffusion_labs.utils.common import DiffusionOutput
from torchmultimodal.modules.layers.normalizations import Fp32GroupNorm
from torchmultimodal.modules.layers.position_embedding import (
    SinusoidalPositionEmbeddings,
)
from torchmultimodal.utils.common import init_module_parameters_to_zero


class LDMUNet(nn.Module):
    """Implements the UNet used by Latent Diffusion Models (LDMs). Composes all
    the blocks for the downsampling encoder, bottleneck, and upsampling encoder in
    the LDMUNet. Constructs the network by adding residual blocks, spatial transformer
    blocks, and up/downsampling blocks for every layer based on user specifications.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752).

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/openaimodel.py#L413

    Overall structure:
    time -> time_embedding -> t
    x, t, context -> encoder -> bottleneck -> decoder -> out

    Attributes:
        in_channels (int): number of input channels.
        model_channels (int): base channel count for the model.
        out_channels (int): number of output channels.
        num_res_blocks_per_level (Union[int, Sequence[int]]): number of residual
            blocks per level. If an integer, then same number of blocks used for
            each level. If a sequence of integers, then the sequence length should
            be same as the length of `channel_multipliers`.
        attention_resolutions (Sequence[int]): sequence of downsampling rates at
            which attention will be performed. For example, if this contains 2,
            then at 2x downsampling, attention will be added in both the down
            and up blocks.
        channel_multipliers (Sequence[int]): list of channel multipliers used by the encoder.
            Decoder uses them in reverse order. Defaults to [1, 2, 4, 8].
        context_dims (Sequence[int], optional): list of dimensions of the conditional
            context tensors. This enables sequential attention support by adding new conditioning
            models to the end of context list. If length of `context_dims` is not the same as
            `num_transformer_layers`, use each element of `context_dims`
            `int(len(context_dims)/num_transformer_layers)` times. Defaults to None.
        use_res_block_updown (bool): if True, use up/down residual blocks for upsampling.
            Defaults to False.
        scale_shift_conditional (bool): if True, splits conditional embedding into two separate
            projections, and adds to hidden state as Norm(h)(w + 1) + b, as described in
            Appendix A in "Improved Denoising Diffusion Probabilistic Models"
            (https://arxiv.org/abs/2102.09672), in resdidual blocks.
            Defaults to False.
        num_attention_heads (int, optional): Number of attention heads used in spatial
            transformer. If None, then `num_channels_per_attention_head` must be provided.
            Defaults to None.
        num_channels_per_attention_head (int, optional): Number of channels for each attention
            head. If None, then `num_heads` must be provided. Defaults to None.
        num_transformer_layers (Union[int, Sequence[int]]): Number of layers in the spatial transformer.
            If an integer, then the same number of layers is used for
            each block. If a sequence of integers, then the sequence length should
            be same as the length of `channel_multipliers`. Defaults to 1.
        use_linear_projection_in_transformer (bool): If True, use linear input and output
            projections in spatial transformer, instead 1x1 convolutions. Defaults to False.
        dropout (float): Dropout value passed to residual blocks. Defaults to 0.0.
        embed_input_size (bool): if True, embed input image size and add to timestep
            embedding. coordinate_embedding_dim must be positive
        embed_target_size (bool): if True, embed target image size and add to timestep
            embedding. coordinate_embedding_dim must be positive
        embed_crop_params (bool): if True, embed crop parameters and add to timestep
            embedding. coordinate_embedding_dim must be positive
        coordinate_embedding_dim (int): embedding dimension for sinusoidal embedding
            of coordinates (e.g. input and target size). If > 0, embed_input_size,
            embed_target_size, or embed_crop_params must be True
        pooled_text_embedding_dim (int): dimension of the pooled text embedding
            to be added to timestep embedding

    Args:
        x (Tensor): input Tensor of shape [b, in_channels, h, w]
        timestep (Tensor): diffusion timesteps of shape [b, ]
        context_list (Sequence[Tensor], optional): Optional list of context Tensors,
            each of shape [b, seq_len_i, context_dim_i]. Defaults to None

    Raises:
        ValueError: If `num_res_blocks_per_level` and `channel_multipliers` do not
            have the same length.
        ValueError: If `num_transformer_layers` and `channel_multipliers` do not
            have the same length.
        ValueError: If both `num_attention_heads` and `num_channels_per_attention_head`
            are None or both are set.
        ValueError: If `model_channels` * `channel_multipliers[0]` is not divisible
            by 32 (number of norm groups).
        RuntimeError: If length of `context_list` in forward is not same as length of `context_dims`.
        RuntimeError: If context tensor ar index `i` does not have embed dim equal to `context_dims[i]`.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks_per_level: Union[int, Sequence[int]],
        attention_resolutions: Sequence[int],
        channel_multipliers: Sequence[int] = (
            1,
            2,
            4,
            8,
        ),
        context_dims: Optional[Sequence[int]] = None,
        use_res_block_updown: bool = False,
        scale_shift_conditional: bool = False,
        num_attention_heads: Optional[int] = None,
        num_channels_per_attention_head: Optional[int] = None,
        num_transformer_layers: Union[int, Sequence[int]] = 1,
        use_linear_projection_in_transformer: bool = False,
        dropout: float = 0.0,
        embed_input_size: bool = False,
        embed_target_size: bool = False,
        embed_crop_params: bool = False,
        coordinate_embedding_dim: int = 0,
        pooled_text_embedding_dim: int = 0,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torchmultimodal.{self.__class__.__name__}")

        num_res_blocks_per_level_list: Sequence[int] = []
        if isinstance(num_res_blocks_per_level, int):
            num_res_blocks_per_level_list = [num_res_blocks_per_level] * len(
                channel_multipliers
            )
        elif isinstance(num_res_blocks_per_level, Sequence):
            num_res_blocks_per_level_list = num_res_blocks_per_level
            if len(num_res_blocks_per_level_list) != len(channel_multipliers):
                raise ValueError(
                    "Expected `num_res_blocks_per_level` to have exactly the same length as `channel_multipliers`"
                    f"({len(channel_multipliers)}), but got {len(num_res_blocks_per_level)}."
                )

        num_transformer_layers_list: Sequence[int] = (
            [num_transformer_layers] * len(channel_multipliers)
            if isinstance(num_transformer_layers, int)
            else num_transformer_layers
        )

        if len(num_transformer_layers_list) != len(channel_multipliers):
            raise ValueError(
                "Expected `num_transformer_layers` to have exactly the same length as `channel_multipliers`"
                f"({len(channel_multipliers)}), but got {len(num_transformer_layers_list)}."
            )

        if num_attention_heads is None and num_channels_per_attention_head is None:
            raise ValueError(
                "Only one of `num_attention_heads` or `num_channels_per_attention_head` "
                "can be set, but none were set."
            )
        elif (
            num_attention_heads is not None
            and num_channels_per_attention_head is not None
        ):
            raise ValueError(
                "Only one of `num_attention_heads` and `num_channels_per_attention_head` can be set,"
                " but both were set."
            )

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks_per_level = num_res_blocks_per_level_list
        self.attention_resolutions = attention_resolutions
        self.channel_multipliers = channel_multipliers
        self.use_res_block_updown = use_res_block_updown
        self.scale_shift_conditional = scale_shift_conditional
        self.dropout = dropout
        self.context_dims = context_dims
        self.num_attention_heads = num_attention_heads
        self.num_channels_per_attention_head = num_channels_per_attention_head
        self.num_transformer_layers = num_transformer_layers_list
        self.use_linear_projection_in_transformer = use_linear_projection_in_transformer

        # time embedding dim is 4 * model_channels to match the original implementation
        self.time_embedding_dim = model_channels * 4
        self.time_embedding = self._create_time_embedding()

        # Additional embeddings to add in for SDXL
        self.pooled_text_embedding_dim = pooled_text_embedding_dim
        assert self.pooled_text_embedding_dim >= 0

        self.embed_input_size = embed_input_size
        self.embed_target_size = embed_target_size
        self.embed_crop_params = embed_crop_params
        # Multiply by 2 to account for (x, y) coordinates of each param
        self.num_coordinates = 2 * (
            self.embed_input_size + self.embed_target_size + self.embed_crop_params
        )
        self.coordinate_embedding_dim = coordinate_embedding_dim
        assert self.coordinate_embedding_dim >= 0
        assert (self.num_coordinates > 0) == (
            self.coordinate_embedding_dim > 0
        ), "Coordinate embedding can only be used when coordinates are provided"

        if self.num_coordinates and self.coordinate_embedding_dim:
            self.pos_embedding = self._create_pos_embedding()
        self.pooled_text_embedding_dim = pooled_text_embedding_dim
        if (
            self.num_coordinates and self.coordinate_embedding_dim
        ) or self.pooled_text_embedding_dim:
            self.add_embedding = self._create_add_embedding()

        # TODO: Add support for label embeddings
        self.down, down_channels, max_resolution = self._create_downsampling_encoder()
        self.bottleneck = self._create_bottleneck(
            down_channels[-1], num_layers=self.num_transformer_layers[-1]
        )
        self.up = self._create_upsampling_decoder(down_channels, max_resolution)
        # input to the output block will have model_channels * channel_multipliers[0] channels
        self.out = self._create_out_block(model_channels * channel_multipliers[0])

    def _create_time_embedding(self) -> nn.Module:
        return nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim=self.model_channels),
            nn.Linear(self.model_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

    def _create_pos_embedding(self) -> nn.Module:
        return SinusoidalPositionEmbeddings(embed_dim=self.coordinate_embedding_dim)

    def _create_add_embedding(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.coordinate_embedding_dim * self.num_coordinates
                + self.pooled_text_embedding_dim,
                self.time_embedding_dim,
            ),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

    def _add_to_time_embedding(
        self, time_embedding: Tensor, additional_embeddings: Dict[str, Tensor]
    ) -> Tensor:
        add_embeddings = []
        if self.pooled_text_embedding_dim:
            assert (
                "pooled_text_embed" in additional_embeddings
            ), "pooled text embedding not provided"
            pooled_text_embed = additional_embeddings["pooled_text_embed"]

            add_embeddings.append(pooled_text_embed)

        if self.num_coordinates and self.coordinate_embedding_dim:
            assert (
                "coordinates" in additional_embeddings
            ), "coordinates to add to timestep embedding not provided"
            assert (
                additional_embeddings["coordinates"].shape[1] == self.num_coordinates
            ), f"Unexpected number of coordinate values: {additional_embeddings['coordinates'].shape[1]} != {self.num_coordinates}"
            coordinates = additional_embeddings["coordinates"].flatten()

            pos_embed = self.pos_embedding(coordinates)
            pos_embed = pos_embed.reshape(
                (
                    time_embedding.shape[0],
                    self.coordinate_embedding_dim * self.num_coordinates,
                )
            )

            add_embeddings.append(pos_embed)

        if add_embeddings:
            add_embed = torch.concat(add_embeddings, dim=-1)
            add_embed = self.add_embedding(add_embed)

            time_embedding = time_embedding + add_embed

        return time_embedding

    def _create_downsampling_encoder(self) -> Tuple[nn.ModuleList, List[int], int]:
        """Returns a nn.ModuleList of downsampling residual blocks, channel count
        for decoder connections and max downsampling rate.
        """
        # Keep track of output channels of every block for thru connections to decoder
        down_channels = []
        # Use ADMStack for conv layer so we can pass in conditional inputs and ignore them
        init_conv = ADMStack()
        init_conv.append_simple_block(
            nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
        )
        down_channels.append(self.model_channels)

        encoder_blocks = nn.ModuleList([init_conv])
        channels_list = tuple(
            [
                self.model_channels * multiplier
                for multiplier in [1] + list(self.channel_multipliers)
            ]
        )
        # keep a track of downsampling rate so we can add attention and
        # return the max downsampling rate
        downsampling_rate = 1
        num_resolutions = len(self.channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = channels_list[level_idx]
            block_out = channels_list[level_idx + 1]
            res_blocks_list, res_block_channels = res_block_adm_stack(
                block_in,
                block_out,
                self.time_embedding_dim,
                self.num_res_blocks_per_level[level_idx],
                self.num_transformer_layers[level_idx],
                self.scale_shift_conditional,
                self.dropout,
                attention_fn=self._create_attention
                if downsampling_rate in self.attention_resolutions
                else None,
            )
            # add residual blocks for each level to encoder blocks
            encoder_blocks.extend(res_blocks_list)
            # add residual block channels for each level to down channels
            down_channels.extend(res_block_channels)

            # add downsampling blocks for all levels except the last one
            if level_idx != num_resolutions - 1:
                downsampling_block = ADMStack()
                # use residual block for downsampling
                if self.use_res_block_updown:
                    downsampling_block.append_residual_block(
                        res_block(
                            block_out,
                            block_out,
                            self.time_embedding_dim,
                            self.scale_shift_conditional,
                            self.dropout,
                            use_downsample=True,
                        )
                    )
                # use conv downsampling
                else:
                    downsampling_block.append_simple_block(
                        Downsample2D(block_out, asymmetric_padding=False)
                    )
                encoder_blocks.append(downsampling_block)
                down_channels.append(block_out)
                # increase downsampling rate for next level
                downsampling_rate *= 2
        return encoder_blocks, down_channels, downsampling_rate

    def _create_upsampling_decoder(
        self, down_channels: List[int], max_downsampling_rate: int
    ) -> nn.ModuleList:
        """
        Args:
            down_channels (List[int]): list of down sample channels for
                through connections from encoder.
            max_downsampling_rate (int): max downsampling rate from the decoder.
        """
        decoder_blocks = nn.ModuleList()
        reversed_channels_list = tuple(
            reversed(
                [
                    self.model_channels * multiplier
                    for multiplier in list(self.channel_multipliers)
                    + [self.channel_multipliers[-1]]
                ]
            )
        )
        reversed_res_blocks_per_level = tuple(reversed(self.num_res_blocks_per_level))
        reversed_num_transformer_layers = tuple(reversed(self.num_transformer_layers))
        downsampling_rate = max_downsampling_rate
        num_resolutions = len(self.channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = reversed_channels_list[level_idx]
            block_out = reversed_channels_list[level_idx + 1]
            res_blocks_list, _ = res_block_adm_stack(
                block_in,
                block_out,
                self.time_embedding_dim,
                # Code ref uses + 1 res blocks in upsampling decoder to add
                # extra res block before upsampling
                reversed_res_blocks_per_level[level_idx] + 1,
                reversed_num_transformer_layers[level_idx],
                self.scale_shift_conditional,
                self.dropout,
                attention_fn=self._create_attention
                if downsampling_rate in self.attention_resolutions
                else None,
                # For through connection from the encoder
                additional_input_channels=down_channels,
            )
            decoder_blocks.extend(res_blocks_list)

            if level_idx != num_resolutions - 1:
                # Key difference between encoder and decoder is that encoder performs
                # downsampling as a separate block whose outputs are saved in forward.
                # On the other hand in the decoder, upsampling and the last residual
                # block are done in a single forward pass operation.
                # Hence in encoder, downsampling is wrapped in its own `ADMStack` while in the
                # decoder, upsampling and final residual block are wrapped in the same `ADMStack`.
                last_res_stack = decoder_blocks[-1]
                if self.use_res_block_updown:
                    last_res_stack.append_residual_block(
                        res_block(
                            block_out,
                            block_out,
                            self.time_embedding_dim,
                            self.scale_shift_conditional,
                            self.dropout,
                            use_upsample=True,
                        )
                    )
                else:
                    last_res_stack.append_simple_block(Upsample2D(block_out))
                downsampling_rate = downsampling_rate // 2
        return decoder_blocks

    def _create_bottleneck(self, channels: int, num_layers: int) -> ADMStack:
        bottleneck = ADMStack()
        bottleneck.append_residual_block(
            res_block(
                channels,
                channels,
                self.time_embedding_dim,
                self.scale_shift_conditional,
                self.dropout,
            )
        )
        bottleneck.append_attention_block(self._create_attention(channels, num_layers))
        bottleneck.append_residual_block(
            res_block(
                channels,
                channels,
                self.time_embedding_dim,
                self.scale_shift_conditional,
                self.dropout,
            )
        )
        return bottleneck

    def _create_out_block(self, channels: int) -> nn.Module:
        conv = nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1)
        # Initialize output projection with zero weight and bias. This helps with
        # training stability. Initialization trick from Fixup Initialization.
        # https://arxiv.org/abs/1901.09321
        init_module_parameters_to_zero(conv)
        return nn.Sequential(
            self._create_norm(channels),
            nn.SiLU(),
            conv,
        )

    def _create_norm(self, channels: int) -> nn.Module:
        # Original LDM implementation hardcodes norm groups to 32
        return Fp32GroupNorm(32, channels)

    def _get_num_attention_heads(self, channels: int) -> int:
        # if num attention heads is not given, then calculate it by
        # dividing channles with num_channels_per_attention_head
        if self.num_channels_per_attention_head is not None:
            return channels // self.num_channels_per_attention_head
        elif self.num_attention_heads is not None:
            return self.num_attention_heads
        # Should never happen. Adding to make Pyre happy
        return 1

    def _create_attention(self, channels: int, num_layers: int) -> nn.Module:
        # original LDM implementation does not pass dropout to SpatialTransformer
        # from UNet, instead just hardcodes it.
        return SpatialTransformer(
            in_channels=channels,
            num_heads=self._get_num_attention_heads(channels),
            num_layers=num_layers,
            context_dims=self.context_dims,
            use_linear_projections=self.use_linear_projection_in_transformer,
        )

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        context_list: Optional[Sequence[Tensor]] = None,
        additional_embeddings: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # Check if context list is provided, then every context embedding has
        # dim as specified in `self.context_dims`.
        if isinstance(context_list, Sequence) and isinstance(
            self.context_dims, Sequence
        ):
            # needed to keep Pyre happy
            context_dims = self.context_dims or []
            if len(context_list) != len(context_dims):
                raise RuntimeError(
                    f"Expected {len(context_dims)} context tensors. Got {len(context_list)}."
                )
            for i in range(len(context_list)):
                if context_list[i].size()[-1] != context_dims[i]:
                    raise RuntimeError(
                        f"Expect context tensor at index {i} to have {context_dims[i]} dim, "
                        f"got dim {context_list[i].size()[-1]}"
                    )
        time_embedding = self.time_embedding(timestep)

        # Add additional conditions to the time embedding if provided
        if additional_embeddings:
            time_embedding = self._add_to_time_embedding(
                time_embedding, additional_embeddings
            )

        h = x
        hidden_states = []
        for block in self.down:
            h = block(h, time_embedding, context_list)
            hidden_states.append(h)
        h = self.bottleneck(h, time_embedding, context_list)
        for block in self.up:
            # through connections from the encoder
            h = torch.cat([h, hidden_states.pop()], dim=1)
            h = block(h, time_embedding, context_list)
        return self.out(h)


def res_block_adm_stack(
    in_channels: int,
    out_channels: int,
    time_embedding_dim: int,
    num_blocks: int,
    num_layers: int,
    scale_shift_conditional: bool,
    dropout: float = 0.0,
    attention_fn: Optional[Callable[[int, int], nn.Module]] = None,
    additional_input_channels: Optional[List[int]] = None,
) -> Tuple[nn.ModuleList, List[int]]:
    """Create a stack of residual blocks wrapped in ADMStack.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        time_embedding_dim (int): dimension of time embeddings
        num_blocks (int): number of residual blocks
        scale_shift_conditional (bool): If True, scale shift conditionals
        dropout (float, optional): dropout rate. Defaults to 0.0.
        attention_fn (Callable[[int], nn.Module], optional): function to be called
            to build the attention module. If None, no attention module is created.
            Defaults to None.
        additional_input_channels (List[int], optional): additional input channels for
            through connections from UNet encoder to decoder. If None, no additional
            input channels are added. Defaults to None.

    Returns:
        nn.ModuleList: list of residual blocks.
        List[int]: list of output channel sizes from each block.
    """
    blocks = nn.ModuleList()
    block_channels = []
    block_in, block_out = in_channels, out_channels
    for _ in range(num_blocks):
        stack = ADMStack()
        stack.append_residual_block(
            res_block(
                block_in
                + (additional_input_channels.pop() if additional_input_channels else 0),
                block_out,
                time_embedding_dim,
                scale_shift_conditional,
                dropout,
            )
        )
        block_in = block_out
        if attention_fn is not None:
            stack.append_attention_block(attention_fn(block_in, num_layers))
        blocks.append(stack)
        block_channels.append(block_out)
    return blocks, block_channels


def res_block(
    in_channels: int,
    out_channels: int,
    time_embedding_dim: int,
    scale_shift_conditional: bool,
    dropout: float,
    use_upsample: bool = False,
    use_downsample: bool = False,
) -> ResBlock:
    """Create one residual block based on parameters.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        time_embedding_dim (int): dimension of time embeddings
        scale_shift_conditional (bool): If True, scale shift conditionals
        dropout (float): dropout rate
        use_upsample (bool): If True, use upsampling in resdiual block
        use_downsample (bool): If True, use downsampling in resdiual block

    Returns:
        ResBlock: Residual block module

    Raises:
        ValueError: If both `use_upsample` and `use_downsample` are True.

    """
    if use_downsample and use_upsample:
        raise ValueError("Cannot use both upsample and downsample in res block")
    res_block_partial = partial(
        ResBlock,
        in_channels=in_channels,
        out_channels=out_channels,
        pre_outconv_dropout=dropout,
        scale_shift_conditional=scale_shift_conditional,
        use_upsample=use_upsample,
        use_downsample=use_downsample,
        cond_proj=adm_cond_proj(
            dim_cond=time_embedding_dim,
            cond_channels=out_channels,
            scale_shift_conditional=scale_shift_conditional,
        ),
    )
    if in_channels != out_channels:
        residual_block = res_block_partial(
            skip_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
    else:
        residual_block = res_block_partial()
    # Initialize residual block's out projection with zero weight and bias.
    # This helps with training stability. Initialization trick from Fixup
    # Initialization : https://arxiv.org/abs/1901.09321
    init_module_parameters_to_zero(residual_block.out_block[-1])
    return residual_block


class LDMModel(nn.Module):
    """Implements the LDM model used by Latent Diffusion Models (LDMs). This is a
    lightweight class that is responsible for composing the LDMUNet and handles
    building the input conditioning tensors from the passed context dictionary. This
    allows us to conveniently use the DDPM, DDIM, CFGuidance, etc modules across
    different models.

    Attributes:
        unet (LDMUNet): Initialized UNet used by the model.
        cond_keys (Sequence[str]): Ordered sequence of conditioning keys to build
            conditional input for cross-attention in unet model. Defaults to tuple().
        additional_cond_keys (Optional[Sequence[str]]): List of conditioning keys to
            be passed as additional conditioning in unet model. These are usually
            projected onto and pooled with the timestep embeddings. Defaults to None.

    Args:
        x (Tensor): input Tensor of shape [b, in_channels, h, w]
        timestep (Tensor): diffusion timesteps of shape [b, ]
        conditional_inputs (Dict[str, Tensor], optional): Optional dictionary of
            context tensors. Key is the conditioning key, value is a tensor of shape
            [b, seq_len, context_dim]. Defaults to None.

    Raises:
        KeyError: If any of the keys in `cond_keys` or `additional_embedding_keys` are
            not present in `conditional_inputs`.
        RuntimeError: If conditional input does not have 3 dims.
    """

    def __init__(
        self,
        unet: LDMUNet,
        cond_keys: Sequence[str] = tuple(),
        additional_cond_keys: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.model = unet
        self.cond_keys = cond_keys
        self.additional_cond_keys = additional_cond_keys

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        conditional_inputs: Optional[Dict[str, Tensor]] = None,
    ):
        context_list = None
        additional_embeddings = None
        if conditional_inputs is not None:
            context_list = [conditional_inputs[k] for k in self.cond_keys]
            for c in context_list:
                if len(c.size()) != 3:
                    raise RuntimeError(
                        f"Expected context tensor to have 3 dims, got {len(c.size())}."
                    )

            if self.additional_cond_keys is not None:
                additional_embeddings = {
                    k: conditional_inputs[k] for k in self.additional_cond_keys
                }

        h = self.model(x, timesteps, context_list, additional_embeddings)
        return DiffusionOutput(prediction=h)
