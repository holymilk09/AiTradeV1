"""BollingerReversion: enters long on a band-low close, exits at midline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
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


def test_long_on_band_low_then_flat_at_midline() -> None:
    strat = BollingerReversion(symbol="TEST", period=10, num_std=1.5)
    # Warm-up: 10 bars at 100 -> band collapses on zero stddev, no signal.
    for i in range(10):
        assert strat.on_bar(_bar(i, 100.0)) is None
    # Inject noise so stddev > 0.
    prices = [101.0, 99.0, 100.5, 99.5, 100.0, 99.8, 100.2, 100.0, 99.9, 100.1]
    for i, p in enumerate(prices, start=10):
        strat.on_bar(_bar(i, p))
    # Sharp drop -> below lower band -> LONG signal.
    sig = strat.on_bar(_bar(20, 95.0))
    assert sig is not None
    assert sig.direction is Direction.LONG
    # Recovery to midline -> FLAT signal.
    sig = strat.on_bar(_bar(21, 100.0))
    assert sig is not None
    assert sig.direction is Direction.FLAT


def test_ignores_other_symbols() -> None:
    strat = BollingerReversion(symbol="TEST", period=5, num_std=1.0)
    other = Bar(
        symbol="OTHER",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )
    assert strat.on_bar(other) is None


def test_period_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="period"):
        BollingerReversion(symbol="X", period=2)
    with pytest.raises(ValueError, match="num_std"):
        BollingerReversion(symbol="X", num_std=0.0)
