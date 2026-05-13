"""DonchianBreakout: enters on N-bar high break, exits on shorter-window low."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar
from aitrade.strategy.examples.donchian_breakout import DonchianBreakout
from aitrade.strategy.signal import Direction


def _bar(i: int, price: float) -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
        open=price,
        high=price + 0.1,
        low=price - 0.1,
        close=price,
        volume=1_000_000,
    )


def test_enters_long_on_breakout() -> None:
    strat = DonchianBreakout(symbol="TEST", entry_period=5, exit_period=3)
    # 5 bars ranging 100..104 — entry window high will be ~104.1
    for i, p in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
        assert strat.on_bar(_bar(i, p)) is None
    # Next bar closes above prior high -> LONG.
    sig = strat.on_bar(_bar(5, 110.0))
    assert sig is not None
    assert sig.direction is Direction.LONG


def test_exits_on_donchian_low_break() -> None:
    strat = DonchianBreakout(symbol="TEST", entry_period=5, exit_period=3)
    # Warm-up + entry
    for i, p in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
        strat.on_bar(_bar(i, p))
    strat.on_bar(_bar(5, 110.0))
    # Three bars holding above exit-low
    for i, p in enumerate([109.0, 108.0, 108.5], start=6):
        strat.on_bar(_bar(i, p))
    # Close below exit-window low -> FLAT.
    sig = strat.on_bar(_bar(9, 100.0))
    assert sig is not None
    assert sig.direction is Direction.FLAT


def test_period_validation() -> None:
    with pytest.raises(ValueError, match="periods"):
        DonchianBreakout(symbol="X", entry_period=1, exit_period=1)
    with pytest.raises(ValueError, match="exit_period"):
        DonchianBreakout(symbol="X", entry_period=10, exit_period=10)
