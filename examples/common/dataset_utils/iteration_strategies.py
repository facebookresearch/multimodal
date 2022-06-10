# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

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

        if "start_idx" in self.config:
            self._current_idx = self.config.start_idx

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


_iteration_strategy_factory_registry: Dict[str, IterationStrategyFactory] = {
    "constant": ConstantIterationStrategyFactory,
    "round_robin": RoundRobinIterationStrategyFactory,
    "random": RandomIterationStrategyFactory,
}


def iteration_strategy_factory(config: DictConfig) -> IterationStrategyFactory:
    return _iteration_strategy_factory_registry[config.type]


DEFAULT_ITERATION_STRATEGY_FACTORY = _iteration_strategy_factory_registry["random"]
