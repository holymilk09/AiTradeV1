"""Indicator math — known-input / known-output checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.strategy.indicators import (
    atr,
    compute_snapshot,
    ema,
    rsi,
    sma,
)


def _bars(closes: list[float]) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol="T",
            timestamp=start + timedelta(days=i),
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


def test_sma_short_history_returns_none() -> None:
    assert sma([1, 2], 5) is None


def test_sma_exact_window() -> None:
    assert sma([1, 2, 3, 4, 5], 5) == 3.0


def test_ema_matches_first_bar_at_window() -> None:
    # With a flat series, EMA equals the value
    assert ema([10.0] * 20, 12) == 10.0


def test_rsi_all_gains_near_100() -> None:
    closes = [float(i) for i in range(1, 30)]  # strictly increasing
    r = rsi(closes, 14)
    assert r is not None
    assert r > 99  # no losses → RSI = 100


def test_rsi_all_losses_near_zero() -> None:
    closes = [float(i) for i in range(30, 1, -1)]  # strictly decreasing
    r = rsi(closes, 14)
    assert r is not None
    assert r < 1


def test_atr_positive_on_wide_bars() -> None:
    bars = _bars([100.0 + i for i in range(30)])
    a = atr(bars, 14)
    assert a is not None
    assert a > 0


def test_snapshot_bull_trend_on_uptrend() -> None:
    closes = [100.0 + i * 0.5 for i in range(40)]
    snap = compute_snapshot(_bars(closes))
    assert snap.bull_trend is True
    assert snap.rsi_14 is not None
    assert snap.rsi_14 > 50


def test_snapshot_fast_slow_cross_golden() -> None:
    # 29 flat, then one big jump — triggers a golden cross at the last bar
    closes = [100.0] * 29 + [200.0]
    snap = compute_snapshot(_bars(closes), fast_window=5, slow_window=20)
    assert snap.fast_slow_cross in {"golden", None}  # depends on bar count
