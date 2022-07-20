# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional

from torch import nn, Tensor
from torchmultimodal.models.late_fusion import LateFusion


class TwoTowerOutput(NamedTuple):
    output: Tensor
    tower_embeddings: Dict[str, Tensor]


class TwoTower(nn.Module):
    """
    A two tower architecture with a pair of late fusion models
    (for now, can be extended) followed by a fusion for output of each tower.
    Args:
        tower_id_to_tower (Dict[str, LateFusion]): mapping of tower id
        to tower model. Size should be 2, same tower should be passed in
        for shared towers
        tower fusion (nn.Module): Module fusing list of tensors (tower outputs)
        into single output
        shared_tower_id_to_channel_mapping (Optional[Dict[str, Dict[str, str]]]): Dict
        of shared tower id to mapping of channel names of the shared tower
         to the original input channel name
    Inputs:
        channel_to_input (Dict[str,Tensor]) : Channel name to input tensor dict
    """

    def __init__(
        self,
        tower_id_to_tower: Dict[str, LateFusion],
        tower_fusion: nn.Module,
        shared_tower_id_to_channel_mapping: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        super().__init__()
        # lets add this validation for now,
        # we can possibly make this a n tower architecture later.
        if len(tower_id_to_tower) != 2:
            raise ValueError(
                f"Two tower needs 2 towers but found \
                {len(tower_id_to_tower)} towers"
            )
        self.tower_id_to_tower = nn.ModuleDict(tower_id_to_tower)
        self.tower_fusion = tower_fusion
        if shared_tower_id_to_channel_mapping is not None:
            towers = list(tower_id_to_tower.values())
            if towers[0] != towers[1]:
                raise ValueError(
                    "Towers should be shared if channel mapping is passed in"
                )
        self.shared_tower_id_to_channel_mapping: Optional[
            Dict[str, Dict[str, str]]
        ] = shared_tower_id_to_channel_mapping

    def forward(self, channel_to_input: Dict[str, Tensor]) -> TwoTowerOutput:
        tower_embeddings = OrderedDict()
        for tower_id, tower in self.tower_id_to_tower.items():
            tower_input = self._get_tower_input(
                tower_id, list(tower.encoders.keys()), channel_to_input
            )
            tower_embeddings[tower_id] = tower(tower_input)

        final_out = self.tower_fusion(list(tower_embeddings.values()))
        return TwoTowerOutput(output=final_out, tower_embeddings=tower_embeddings)

    def _get_tower_input(
        self,
        tower_id: str,
        tower_channels: List[str],
        channel_to_input: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        tower_input = {}
        channel_name_mapping: Dict[str, str] = {}
        if self.shared_tower_id_to_channel_mapping is not None:
            if self.shared_tower_id_to_channel_mapping.get(tower_id) is not None:
                channel_name_mapping = self.shared_tower_id_to_channel_mapping[tower_id]

        for channel in tower_channels:
            if channel_name_mapping.get(channel) is not None:
                input_channel_name = channel_name_mapping[channel]
            else:
                input_channel_name = channel
            tower_input[channel] = channel_to_input[input_channel_name]
        return tower_input
