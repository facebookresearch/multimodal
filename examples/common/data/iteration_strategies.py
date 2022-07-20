# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class IterationStrategy:
    """
    Base class for defining iteration strategies that will be used
    for iterating over multiple datasets.

    An IterationStrategy implementation `__call__` method
    which returns index of dataset from which next batch must be
    pulled.

    Args:
        config (DictConfig): Object of type DictConfig which contains configuration parameters.
        dataloaders (Dict[str, DataLoader]): A dictionary containing mapping from dataset key
            to its dataloader.
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        self.config = config
        self.dataloaders = dataloaders

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ hasn't been implemented")


class IterationStrategyFactory:
    def __init__(self, config: DictConfig):
        self.config = config

    def __call__(
        self, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ) -> IterationStrategy:
        raise NotImplementedError("__call__ hasn't been implemented")


class ConstantIterationStrategy(IterationStrategy):
    """
    Always returns a constant number. Can be used for single dataset training.

    Config Parameters:
        idx: index to be returned
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._idx = self.config.idx

    def __call__(self, *args, **kwargs):
        return self._idx


class ConstantIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return ConstantIterationStrategy(self.config, dataloaders)


class RoundRobinIterationStrategy(IterationStrategy):
    """
    Samples datasets one by one in round robin fashion.

    Config Parameters:
        start_idx: index of the dataset to be returned first
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._current_idx = self.config.start_idx if "start_idx" in self.config else 0

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.dataloaders)
        return nxt


class RoundRobinIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return RoundRobinIterationStrategy(self.config, dataloaders)


class RandomIterationStrategy(IterationStrategy):
    """
    Samples random number each time when sampled.
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(len(self.dataloaders), 1)[0]
        return choice


class RandomIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return RandomIterationStrategy(self.config, dataloaders)


class SizeProportionalIterationStrategy(IterationStrategy):
    """
    Samples index based on size of each dataset. Bigger datasets are sampled more.
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._per_dataloader_lengths = []
        self._total_length = 0

        for key, loader in self.dataloaders.items():
            n = len(loader)
            self._per_dataloader_lengths.append(n)
            self._total_length += n

        self._dataloader_probabilities = self._per_dataloader_lengths
        self._dataloader_probabilities = [
            prob / self._total_length for prob in self._dataloader_probabilities
        ]

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataloader_probabilities
        )[0]
        return choice


class SizeProportionalIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return SizeProportionalIterationStrategy(self.config, dataloaders)


class RatiosIterationStrategy(IterationStrategy):
    """
    Samples based on ratios specified as `sampling_ratios` parameter in the config.

    Config Parameters:
        sampling_ratios: defines a dictionary pointing from dataset key to a floating ration
            specifying how much the dataset should be sampled. Floats together should sum to one.
    """

    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        sampling_ratios = self.config.params.get("sampling_ratios", {})
        probabilities = []
        for data_key in dataloaders.keys():
            assert (
                data_key in sampling_ratios
            ), f"{data_key} must be specified in sampling_ratios param for multitasking {sampling_ratios}"
            probabilities.append(sampling_ratios[data_key])

        # normalize the sampling ratios to sum up to 1
        prob_sum = sum(probabilities)
        assert all(prob >= 0 for prob in probabilities) and prob_sum > 0, (
            "sampling_ratios param for multitasking must be all non-negative "
            "and at least one of them needs to be positive."
        )

        self._probabilities = [prob / prob_sum for prob in probabilities]

    def __call__(self, *args, **kwargs):
        return np.random.choice(len(self.dataloaders), 1, p=self._probabilities)[0]


class RatiosIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return RatiosIterationStrategy(self.config, dataloaders)


_iteration_strategy_factory_registry: Dict[str, IterationStrategyFactory] = {
    "constant": ConstantIterationStrategyFactory,
    "round_robin": RoundRobinIterationStrategyFactory,
    "random": RandomIterationStrategyFactory,
    "size_proportional": SizeProportionalIterationStrategyFactory,
    "ratios": RatiosIterationStrategyFactory,
}


def iteration_strategy_factory(config: DictConfig) -> IterationStrategyFactory:
    return _iteration_strategy_factory_registry[config.type](config)


DEFAULT_ITERATION_STRATEGY_FACTORY = _iteration_strategy_factory_registry[
    "round_robin"
](DictConfig(content={}))
