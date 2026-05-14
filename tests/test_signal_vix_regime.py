"""VixRegimeScorer — realized-vol regime conviction premium."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.signals.components import ScoringContext, VixRegimeScorer


def _bars_with_vol(n: int, daily_vol: float, seed: int = 0) -> list[Bar]:
    """Synthetic series with prescribed *daily* volatility.

    Test-only — never feed this to a backtest. Used here purely to
    construct a deterministic high/low-vol regime to verify the scorer
    classifies the *current* window correctly.
    """
    rng = random.Random(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0
    out: list[Bar] = []
    for i in range(n):
        r = rng.gauss(0.0, daily_vol)
        price *= math.exp(r)
        out.append(
            Bar(
                symbol="TEST",
                timestamp=start + timedelta(days=i),
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=1_000_000,
            )
        )
    return out


def test_insufficient_history_returns_neutral() -> None:
    scorer = VixRegimeScorer()
    bars = _bars_with_vol(20, 0.01)
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.score == 0.5
    assert "insufficient" in c.evidence


def test_low_vol_window_following_high_vol_history() -> None:
    """Build 200 high-vol bars, then 30 low-vol bars at the end. The
    scorer's *current* 20-day rvol is in the low-vol tail of the 252d
    distribution → should score near 1.0 ("good overnight regime")."""
    scorer = VixRegimeScorer()
    bars = _bars_with_vol(200, 0.03, seed=1) + _bars_with_vol(30, 0.003, seed=2)
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.score >= 0.8, f"expected low-regime score near 1.0, got {c.score}"
    assert c.features["rvol_pct"] <= 25.0


def test_high_vol_window_following_low_vol_history() -> None:
    """200 low-vol bars then 30 high-vol bars → current rvol is in the
    top of the distribution → score near 0.0."""
    scorer = VixRegimeScorer()
    bars = _bars_with_vol(200, 0.003, seed=3) + _bars_with_vol(30, 0.05, seed=4)
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.score <= 0.2, f"expected high-regime score near 0.0, got {c.score}"
    assert c.features["rvol_pct"] >= 75.0


def test_features_are_populated() -> None:
    scorer = VixRegimeScorer()
    bars = _bars_with_vol(100, 0.02, seed=5)
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert "rvol_20d" in c.features
    assert "rvol_pct" in c.features
    assert 0.0 < c.features["rvol_20d"] < 2.0
    assert 0.0 <= c.features["rvol_pct"] <= 100.0


def test_weight_is_default() -> None:
    scorer = VixRegimeScorer()
    bars = _bars_with_vol(100, 0.02, seed=6)
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.weight == VixRegimeScorer.default_weight == 0.2
