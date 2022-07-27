# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm


label_color_map = {
    0: torch.FloatTensor((0, 0, 0)),
    1: torch.FloatTensor((128, 0, 0)),
    2: torch.FloatTensor((255, 0, 0)),
    3: torch.FloatTensor((139, 69, 19)),
    4: torch.FloatTensor((0, 255, 0)),
    5: torch.FloatTensor((0, 128, 0)),
    6: torch.FloatTensor((0, 100, 0)),
    7: torch.FloatTensor((244, 164, 96)),
    8: torch.FloatTensor((205, 133, 63)),
    9: torch.FloatTensor((255, 192, 203)),
    10: torch.FloatTensor((210, 105, 30)),
    11: torch.FloatTensor((255, 0, 255)),
    12: torch.FloatTensor((230, 230, 250)),
    13: torch.FloatTensor((0, 191, 255)),
    14: torch.FloatTensor((154, 205, 50)),
    15: torch.FloatTensor((255, 215, 0)),
    16: torch.FloatTensor((169, 169, 169)),
    17: torch.FloatTensor((148, 0, 211)),
    18: torch.FloatTensor((127, 255, 212)),
    19: torch.FloatTensor((255, 255, 0)),
    20: torch.FloatTensor((255, 69, 0)),
    21: torch.FloatTensor((255, 255, 255)),
    22: torch.FloatTensor((0, 0, 255)),
}


def convert_grayscale_to_color_label(input_tensor):
    b_in, t_in, h_in, w_in = input_tensor.shape

    input_tensor = input_tensor.reshape(-1)
    output_tensor = torch.zeros(input_tensor.shape[0], 3)
    for i, t in tqdm(
        enumerate(input_tensor.cpu().numpy()), total=input_tensor.shape[0]
    ):
        output_tensor[i] = label_color_map[t]

    output_tensor = output_tensor.reshape(b_in, t_in, h_in, w_in, 3).permute(
        0, 4, 1, 2, 3
    )

    return output_tensor
