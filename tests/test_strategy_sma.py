"""SMA crossover example emits expected direction changes."""

from __future__ import annotations

from aitrade.data.models import Bar
from aitrade.strategy.examples.sma_crossover import SmaCrossover
from aitrade.strategy.signal import Direction


def test_emits_long_on_trend_up(synthetic_bars: list[Bar]) -> None:
    strat = SmaCrossover(symbol="TEST", fast=5, slow=20)
    signals = [strat.on_bar(b) for b in synthetic_bars]
    non_none = [s for s in signals if s is not None]
    # First real signal should be LONG once the fast MA crosses above the slow MA.
    assert any(s.direction is Direction.LONG for s in non_none)


def test_validates_window_sizes() -> None:
    import pytest

    with pytest.raises(ValueError, match="fast must be < slow"):
        SmaCrossover(symbol="X", fast=20, slow=10)
