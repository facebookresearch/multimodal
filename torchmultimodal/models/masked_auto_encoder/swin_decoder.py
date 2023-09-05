# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Tuple

import torch

from torch import nn, Tensor
from torchmultimodal.modules.layers.mlp import MLP
from torchmultimodal.modules.layers.transformer import TransformerOutput
from torchvision.ops import StochasticDepth

# This custom implementation of swin components for audio mae has been taken from
# https://github.com/facebookresearch/AudioMAE/blob/main/timm_patch/swin_transformer.py


class WindowMultiHeadAttention(nn.Module):
    """
    Window based attention as used by swin v2 https://arxiv.org/pdf/2111.09883.pdf

    Args:
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        window_size (Tuple[int, int]): dimension of the window for local attention.
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        proj_dropout (float): dropout probability for attention output projection. Defaults to 0.
        meta_hidden_dim (int): hidden dim for the mlp for relative position bias. Default is 384.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        meta_hidden_dim: int = 384,
        meta_mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_heads = num_heads

        self.qkv = nn.Linear(in_features=input_dim, out_features=input_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.meta_mlp = MLP(
            in_dim=2,  # x, y
            hidden_dims=meta_hidden_dim,
            out_dim=num_heads,
            activation=nn.ReLU,
            dropout=meta_mlp_dropout,
        )
        self.register_parameter("tau", nn.Parameter(torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        device = self.tau.device
        # 2 x window_area
        coordinates = torch.stack(
            torch.meshgrid(
                [
                    torch.arange(self.window_size[0], device=device),
                    torch.arange(self.window_size[1], device=device),
                ]
            ),
            dim=0,
        ).flatten(1)

        # 2 x window_area x window_area
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]

        # window_area ^ 2 x 2
        relative_coordinates = (
            relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        )

        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs()
        )
        self.register_buffer(
            "relative_coordinates_log", relative_coordinates_log, persistent=False
        )

    def _relative_positional_encodings(self) -> Tensor:
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): input to the attention block of shape bsz x seq_len x input_dim.
            seq_len should match number of patches in the window
            mask (Optional[Tensor]): attention mask of shape total_num_window x seq_len x seq_len. Defaults to None.

        Returns:
            Tensor of shape bsz x seq_len x input_dim
        """
        bsz, seq_len, embed_dim = x.shape
        if seq_len != self.window_size[0] * self.window_size[1]:
            raise ValueError(
                f"Input sequence length {seq_len} needs to match window area"
            )

        qkv = (
            self.qkv(x)
            .view(bsz, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # bsz x num_heads x seq_len x (embed_dim//num_heads)
        query, key, value = qkv.unbind(0)

        # compute scaled cosine attention
        denom = torch.linalg.vector_norm(
            query, dim=-1, keepdim=True
        ) @ torch.linalg.vector_norm(key, dim=-1, keepdim=True).transpose(-2, -1)

        # bsz x num_heads x seq_len x seq_len
        attn = query @ key.transpose(-2, -1) / denom.clamp(min=1e-6)

        attn = attn / self.tau.clamp(min=0.01).reshape(1, self.num_heads, 1, 1)
        attn = attn + self._relative_positional_encodings()

        if mask is not None:
            # mask shape : num_window x seq_len x seq_len
            num_win: int = mask.shape[0]
            attn = attn.view(bsz // num_win, num_win, self.num_heads, seq_len, seq_len)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, seq_len, seq_len)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(bsz, seq_len, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin transformer block customized for audio mae and loosely following swin v2 https://arxiv.org/pdf/2111.09883.pdf

    Args:
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        input_size (Tuple[int, int]): dimension of the original input before patchification
        window_size (Tuple[int, int]): dimension of the window for local attention
        feedforward_dim (int): size of hidden dimension of feedforward network in transformer block
        shift_size (Tuple[int, int]): dimension of shift to be applied to the window. Defaults to (0, 0)
        mlp_dropout (float): dropout probability for mlp in transformer block and projection in SA block.
            Defaults to 0
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        drop_path (float): Drop path probability in transformer. Defaults to 0.0.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-5.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        input_size: Tuple[int, int],
        window_size: Tuple[int, int],
        feedforward_dim: int,
        shift_size: Tuple[int, int] = (0, 0),
        mlp_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.window_size, self.shift_size = self._get_effective_window_shift(
            window_size, shift_size
        )
        self.num_heads = num_heads

        self.attn = WindowMultiHeadAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            window_size=self.window_size,  # type: ignore
            attn_dropout=attn_dropout,
            proj_dropout=mlp_dropout,
        )
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.drop_path1 = (
            StochasticDepth(drop_path, "row") if drop_path > 0.0 else nn.Identity()
        )

        self.mlp = MLP(
            in_dim=input_dim,
            hidden_dims=feedforward_dim,
            dropout=mlp_dropout,
            out_dim=input_dim,
            activation=nn.GELU,
        )
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.drop_path2 = (
            StochasticDepth(drop_path, "row") if drop_path > 0.0 else nn.Identity()
        )

        self._make_attention_mask()

    def _get_effective_window_shift(
        self, target_window_size: Tuple[int, int], target_shift_size: Tuple[int, int]
    ) -> Tuple[Tuple[int, ...], Tuple[Any, ...]]:
        # if input is smaller than window, effective window size is the input size
        window_size: List[int] = [
            f if f <= w else w for f, w in zip(self.input_size, target_window_size)
        ]

        # if input is smaller than window, no need for shift
        shift_size = [
            0 if f <= w else s
            for f, w, s in zip(self.input_size, window_size, target_shift_size)
        ]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) -> None:
        # mask only needed for shift case
        if any(self.shift_size):
            input_h, input_w = self.input_size
            img_mask = torch.zeros((1, input_h, input_w, 1))
            cnt = 0
            for h in (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            ):
                for w in (
                    slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None),
                ):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # num_windows x window_h x window_w x 1
            mask_windows = self._window_partition(img_mask)
            window_area = self.window_size[0] * self.window_size[1]

            mask_windows = mask_windows.view(-1, window_area)

            # num_windows x window_area x window_area
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _window_partition(self, x: Tensor) -> Tensor:
        bsz, h, w, channels = x.shape
        window_h, window_w = self.window_size
        x = x.view(
            bsz,
            h // window_h,
            window_h,
            w // window_w,
            window_w,
            channels,
        )

        # bsz x (h/window_h) x window_h x (w/window_w) x window_w x channel => bsz * num_windows x window_h x window_w x channel
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_h, window_w, channels)
        )
        return windows

    def _window_reverse(self, windows: Tensor, bsz: int) -> Tensor:
        input_h, input_w = self.input_size
        window_h, window_w = self.window_size
        x = windows.view(
            bsz,
            input_h // window_h,
            input_w // window_w,
            window_h,
            window_w,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, input_h, input_w, -1)
        return x

    def _shifted_window_attn(self, x: Tensor) -> Tensor:
        h, w = self.input_size
        bsz, seq_len, channels = x.shape
        if seq_len != h * w:
            raise ValueError(
                f"Input sequence length {seq_len} needs to match input size"
            )
        x = x.view(bsz, h, w, channels)

        # cyclic shift
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

        # partition windows
        x_windows = self._window_partition(x)
        # bsz * num_windows x window_h x window_w x channels => bsz * num_windows x num_window_patches x channels
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1], channels
        )

        # window multihead self attn => bsz * num_windows x window_h x window_w x channels
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size[0], self.window_size[1], channels
        )
        x = self._window_reverse(attn_windows, bsz)

        # reverse cyclic shift
        if do_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))

        x = x.view(bsz, seq_len, channels)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input to the transformer block of shape bsz x seq_len x input_dim.
            seq_len shoud match number of patches as per input_size

        Returns:
            Tensor of shape bsz x seq_len x input_dim
        """
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Swin transformer that stacks layers of the swin block

    Args:
        n_layer (int): number of layers
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        input_size (Tuple[int, int]): dimension of the original input before patchification
        window_size (Tuple[int, int]): dimension of the window for local attention
        feedforward_dim (int): size of hidden dimension of feedforward network in transformer block
        shift_size (Tuple[int, int]): dimension of shift to be applied to the window. Defaults to (0, 0)
        mlp_dropout (float): dropout probability for mlp in transformer block and projection in SA block.
            Defaults to 0
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        drop_path (float): Drop path probability in transformer. Defaults to 0.0.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-5.
        final_layer_norm_eps (float): the eps value in layer norms. Default is 1e-5
    """

    def __init__(
        self,
        n_layer: int,
        input_dim: int,
        num_heads: int,
        input_size: Tuple[int, int],
        window_size: Tuple[int, int],
        feedforward_dim: int,
        mlp_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-5,
        final_layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        layers = []
        for idx in range(n_layer):
            if (idx % 2) == 0:
                shift_size = (0, 0)
            else:
                shift_size = (2, 0)
            layers.append(
                SwinTransformerBlock(
                    input_dim=input_dim,
                    num_heads=num_heads,
                    input_size=input_size,
                    window_size=window_size,
                    shift_size=shift_size,
                    feedforward_dim=feedforward_dim,
                    mlp_dropout=mlp_dropout,
                    attn_dropout=attn_dropout,
                    drop_path=drop_path,
                    layer_norm_eps=layer_norm_eps,
                )
            )

        self.layers = nn.ModuleList(layers)
        self.final_layer_norm = nn.LayerNorm(input_dim, eps=final_layer_norm_eps)

    def forward(self, x: Tensor) -> TransformerOutput:
        """
        Args:
            x (Tensor): input to the transformer block of shape bsz x seq_len x input_dim.
            seq_len shoud match number of patches as per input_size

        Returns:
            Output of type TransformerOutput with last_hidden_state field contain tensor of shape bsz x seq_len x input_dim
            representing output from final layer
        """
        hidden_states = x
        for layer_module in self.layers:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs

        hidden_states = self.final_layer_norm(hidden_states)

        return TransformerOutput(
            last_hidden_state=hidden_states,
        )
