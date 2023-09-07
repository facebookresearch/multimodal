# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn, Tensor


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss that computed MSE between predicted and target values as described in MAE paper
    https://arxiv.org/abs/2111.06377. Loss is averaged only over masked patches.

    Args:
        normalize_target (bool) : Whether target should be normalized. Defaults to True

    """

    def __init__(self, normalize_target: bool = True):
        super().__init__()
        self.normalize_target = normalize_target

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred (Tensor): predicted tensor with shape bsz x num_patches x (patch_size ** 2 * channels)
            target (Tensor): patchified input with the same shape as pred
            mask (Tensor):  Tensor of shape bsz x num_patches indicating which patches are masked.
            1 indicates masked patch amd 0 indicated unmasked patch.
        Returns: computed loss

        """
        if mask.sum() == 0:
            raise ValueError("At least one patch must be masked")

        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `float`.
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
