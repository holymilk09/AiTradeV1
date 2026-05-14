"""VixGatedBollinger — entry gating via VixRegimeScorer.

Strategy is a thin wrapper around BollingerReversion (independently tested
elsewhere) and VixRegimeScorer (independently tested). These tests cover
the *wrapper-specific* behavior: the gate path itself.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar
from aitrade.signals.components import ScoringContext
from aitrade.signals.conviction import ConvictionComponent
from aitrade.strategy.examples.vix_gated_bollinger import VixGatedBollinger
from aitrade.strategy.signal import Direction


class _FixedScorer:
    """Test-only scorer that returns a fixed rvol_pct, ignoring bars."""

    name = "fixed"
    default_weight = 0.2

    def __init__(self, rvol_pct: float) -> None:
        self.rvol_pct = rvol_pct

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        return ConvictionComponent(
            score=1.0 if self.rvol_pct <= 20 else 0.0,
            weight=self.default_weight,
            evidence="test fixture",
            features={"rvol_pct": self.rvol_pct, "rvol_20d": 0.15},
        )


def _trending_then_drop(period: int = 20) -> list[Bar]:
    """`period` rising closes, then one big drop below the lower band.

    Deterministic; no RNG. Inner BollingerReversion will see a stable
    series above its mid, then a sharp close below mid - 2σ on the final
    bar — a clean entry trigger.
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i in range(period):
        price = 100.0 + i * 0.05  # tiny drift, very low realized vol
        bars.append(Bar(
            symbol="TEST",
            timestamp=start + timedelta(days=i),
            open=price, high=price + 0.01, low=price - 0.01,
            close=price, volume=1_000_000,
        ))
    # Drop bar — close 5% below the mid.
    drop_price = 100.0 * 0.95
    bars.append(Bar(
        symbol="TEST",
        timestamp=start + timedelta(days=period),
        open=drop_price, high=drop_price + 0.5, low=drop_price - 0.5,
        close=drop_price, volume=1_000_000,
    ))
    return bars


def test_gate_permits_entry_when_pct_below_threshold() -> None:
    strat = VixGatedBollinger(symbol="TEST", period=20, num_std=2.0, rvol_pct_max=50.0)
    strat._scorer = _FixedScorer(rvol_pct=15.0)  # well below threshold
    bars = _trending_then_drop(period=20)
    signal = None
    for b in bars:
        signal = strat.on_bar(b)
    assert signal is not None, "low-vol regime should permit entry"
    assert signal.direction is Direction.LONG


def test_gate_blocks_entry_when_pct_above_threshold() -> None:
    strat = VixGatedBollinger(symbol="TEST", period=20, num_std=2.0, rvol_pct_max=50.0)
    strat._scorer = _FixedScorer(rvol_pct=85.0)  # well above threshold
    bars = _trending_then_drop(period=20)
    signal = None
    for b in bars:
        signal = strat.on_bar(b)
    assert signal is None, "high-vol regime should block entry"


def test_exit_signal_always_passes_through() -> None:
    """Set the strategy LONG via an entry in low-vol regime, then flip
    the gate to high-vol and produce an exit. Exit must pass even though
    the gate is closed for entries."""
    strat = VixGatedBollinger(symbol="TEST", period=20, num_std=2.0, rvol_pct_max=50.0)
    strat._scorer = _FixedScorer(rvol_pct=10.0)  # gate open
    bars = _trending_then_drop(period=20)
    for b in bars:
        strat.on_bar(b)
    # Confirm we're long.
    assert strat._inner._last_direction is Direction.LONG

    # Flip gate closed — exits must still pass.
    strat._scorer = _FixedScorer(rvol_pct=95.0)
    # Construct an exit bar: close back at or above the inner mid.
    cur_mid = sum(strat._inner._closes) / len(strat._inner._closes)
    exit_bar = Bar(
        symbol="TEST",
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
        open=cur_mid * 1.02, high=cur_mid * 1.03,
        low=cur_mid * 1.0, close=cur_mid * 1.02,
        volume=1_000_000,
    )
    signal = strat.on_bar(exit_bar)
    assert signal is not None, "exits must always pass"
    assert signal.direction is Direction.FLAT


def test_rvol_pct_max_validation() -> None:
    with pytest.raises(ValueError, match="rvol_pct_max"):
        VixGatedBollinger(symbol="X", rvol_pct_max=150.0)
    with pytest.raises(ValueError, match="rvol_pct_max"):
        VixGatedBollinger(symbol="X", rvol_pct_max=-1.0)


def test_bar_history_is_bounded() -> None:
    """Memory bound: feeding 400 bars keeps the internal history at the cap."""
    strat = VixGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rng = random.Random(42)
    for i in range(400):
        p = 100.0 * math.exp(rng.gauss(0, 0.01))
        strat.on_bar(Bar(
            symbol="TEST",
            timestamp=start + timedelta(days=i),
            open=p, high=p * 1.001, low=p * 0.999, close=p, volume=1_000_000,
        ))
    assert len(strat._bars) <= 300
