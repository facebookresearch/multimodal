# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#
# preprocess.py
#
# This file contains utility functions to process, reshape, etc. Tensors
#
#


def flatten_to_channel_vectors(x, channel_dim):
    """
    Takes an input tensor and flattens across all dimensions except the specified dimension
    An example is flattening an encoder output volume of BxCxHxW to (B*H*W)xC for a VQVAE model
    """

    # Move channel dim to end
    new_dims = tuple(
        [i for i in range(len(x.shape)) if i != channel_dim] + [channel_dim]
    )
    x = x.permute(new_dims).contiguous()
    permuted_shape = x.shape

    # Flatten input
    x_flat = x.view(-1, permuted_shape[-1])

    return x_flat, permuted_shape


def reshape_from_channel_vectors(x_flat, permuted_shape, orig_channel_dim):
    """
    The inverse of flatten_to_channel_vectors
    """
    # Reshape flattened vectors to permuted shape
    x_out = x_flat.view(permuted_shape)

    # Move channel dim (last) back to original spot
    old_dims = list(range(len(permuted_shape)))
    old_dims.pop(-1)
    old_dims.insert(orig_channel_dim, len(permuted_shape) - 1)
    x_out = x_out.permute(old_dims).contiguous()

    return x_out
