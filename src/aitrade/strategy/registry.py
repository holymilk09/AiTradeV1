"""Strategy name → factory registry used by the CLI."""

from __future__ import annotations

from collections.abc import Callable

from aitrade.strategy.base import Strategy
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.examples.donchian_breakout import DonchianBreakout
from aitrade.strategy.examples.sma_crossover import SmaCrossover
from aitrade.strategy.examples.vix_gated_bollinger import VixGatedBollinger

StrategyFactory = Callable[..., Strategy]


_REGISTRY: dict[str, StrategyFactory] = {
    "sma_crossover": SmaCrossover,
    "bollinger_reversion": BollingerReversion,
    "donchian_breakout": DonchianBreakout,
    "vix_gated_bollinger": VixGatedBollinger,
}


def get_strategy(name: str, **kwargs: object) -> Strategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def register_strategy(name: str, factory: StrategyFactory) -> None:
    _REGISTRY[name] = factory


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)
