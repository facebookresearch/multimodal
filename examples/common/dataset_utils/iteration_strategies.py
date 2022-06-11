# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class IterationStrategy:
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
    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        # self._check_not_epoch_training()

        self._current_idx = self.config.start_idx if "start_idx" in self.config else 0

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.dataloaders)
        return nxt


class RoundRobinIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return RoundRobinIterationStrategy(self.config, dataloaders)


class RandomIterationStrategy(IterationStrategy):
    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._check_not_epoch_training()

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(len(self.dataloaders), 1)[0]
        return choice


class RandomIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return RandomIterationStrategy(self.config, dataloaders)


class SizeProportionalIterationStrategy(IterationStrategy):
    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        self._per_dataset_lengths = []
        self._total_length = 0

        for loader in self.dataloaders.values():
            self._per_dataset_lengths.append(len(loader))
            self._total_length += dataset_instance_length

        self._dataset_probabilities = self._per_dataset_lengths[:]
        self._dataset_probabilities = [
            prob / self._total_length for prob in self._dataset_probabilities
        ]

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataset_probabilities
        )[0]
        return choice


class SizeProportionalIterationStrategyFactory(IterationStrategyFactory):
    def __call__(self, dataloaders: Dict[str, DataLoader], *args, **kwargs):
        return SizeProportionalIterationStrategy(self.config, dataloaders)


class RatiosIterationStrategy(IterationStrategy):
    def __init__(
        self, config: DictConfig, dataloaders: Dict[str, DataLoader], *args, **kwargs
    ):
        super().__init__(config, dataloaders, *args, **kwargs)
        # self._check_not_epoch_training()
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
        choice = np.random.choice(len(self.dataloaders), 1, p=self._probabilities)[0]
        return choice


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


DEFAULT_ITERATION_STRATEGY_FACTORY = _iteration_strategy_factory_registry["random"]
