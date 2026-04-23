"""Strategy name → factory registry used by the CLI."""

from __future__ import annotations

from collections.abc import Callable

from aitrade.strategy.base import Strategy
from aitrade.strategy.examples.sma_crossover import SmaCrossover

StrategyFactory = Callable[..., Strategy]


_REGISTRY: dict[str, StrategyFactory] = {
    "sma_crossover": SmaCrossover,
}


def get_strategy(name: str, **kwargs: object) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_strategy(name: str, factory: StrategyFactory) -> None:
    _REGISTRY[name] = factory


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)
