"""VwapEmaVolume — VWAP reclaim + EMA stack + volume confirmation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar
from aitrade.strategy.examples.vwap_ema_volume import VwapEmaVolume
from aitrade.strategy.signal import Direction


def _bar(i: int, *, close: float, volume: float = 1_000_000.0,
         spread: float = 0.5) -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
        open=close,
        high=close + spread,
        low=close - spread,
        close=close,
        volume=volume,
    )


def test_param_validation() -> None:
    with pytest.raises(ValueError, match="vwap_window"):
        VwapEmaVolume(symbol="X", vwap_window=3)
    with pytest.raises(ValueError, match="ema_fast"):
        VwapEmaVolume(symbol="X", ema_fast=21, ema_slow=8)
    with pytest.raises(ValueError, match="min_volume_ratio"):
        VwapEmaVolume(symbol="X", min_volume_ratio=0.0)


def test_no_signal_until_history_warmed() -> None:
    strat = VwapEmaVolume(symbol="TEST")
    for i in range(15):
        assert strat.on_bar(_bar(i, close=100.0 + i * 0.1)) is None


def test_entry_when_reclaim_and_uptrend_and_volume() -> None:
    """Build a series that breaks below VWAP for a stretch, then reclaims
    with confirmed volume + the EMA stack already bullish."""
    strat = VwapEmaVolume(symbol="TEST", vwap_window=20, ema_fast=8, ema_slow=21,
                          vol_avg_window=20, min_volume_ratio=1.3)
    # 25 calm-uptrend bars to set the EMA stack bullish (fast > slow).
    closes = [100.0 + i * 0.3 for i in range(25)]
    # 5 dip bars below VWAP — drop below the running VWAP for a few sessions.
    closes += [c - 4.0 for c in closes[-5:]]
    # 1 reclaim bar — pop sharply above with high volume.
    closes.append(closes[-1] + 8.0)
    sig = None
    for i, c in enumerate(closes[:-1]):
        sig = strat.on_bar(_bar(i, close=c, volume=1_000_000))
    # Last bar: high-volume reclaim
    sig = strat.on_bar(
        _bar(len(closes) - 1, close=closes[-1], volume=2_000_000, spread=2.0)
    )
    assert sig is not None
    assert sig.direction is Direction.LONG
    assert "vwap_reclaim" in sig.reason


def test_no_entry_without_volume_confirmation() -> None:
    """Same setup as the entry-passing test, but normal volume on the
    reclaim bar — should NOT fire."""
    strat = VwapEmaVolume(symbol="TEST", min_volume_ratio=1.5)
    closes = [100.0 + i * 0.3 for i in range(25)]
    closes += [c - 4.0 for c in closes[-5:]]
    closes.append(closes[-1] + 8.0)
    for i, c in enumerate(closes[:-1]):
        strat.on_bar(_bar(i, close=c, volume=1_000_000))
    # Reclaim with average volume (ratio ≈ 1.0).
    sig = strat.on_bar(
        _bar(len(closes) - 1, close=closes[-1], volume=900_000, spread=2.0)
    )
    assert sig is None


def test_exit_when_close_drops_below_fast_ema() -> None:
    """Get the strategy long, then feed a bar that closes well below the
    8-EMA — should emit FLAT."""
    strat = VwapEmaVolume(symbol="TEST", min_volume_ratio=1.0)
    # Run an entry sequence: uptrend + dip + high-volume reclaim.
    closes = [100.0 + i * 0.3 for i in range(25)]
    closes += [c - 4.0 for c in closes[-5:]]
    closes.append(closes[-1] + 8.0)
    for i, c in enumerate(closes[:-1]):
        strat.on_bar(_bar(i, close=c, volume=1_000_000))
    strat.on_bar(_bar(len(closes) - 1, close=closes[-1], volume=2_500_000))
    assert strat._last_direction is Direction.LONG
    # Now feed a bar that closes deep below the 8-EMA.
    sig = strat.on_bar(_bar(len(closes), close=closes[-1] - 12.0,
                            volume=1_000_000))
    assert sig is not None
    assert sig.direction is Direction.FLAT


def test_exit_when_ema_cross_down() -> None:
    """Long, then EMA stack flips (fast crosses below slow). Emit FLAT."""
    strat = VwapEmaVolume(symbol="TEST", min_volume_ratio=1.0)
    closes = [100.0 + i * 0.3 for i in range(25)]
    closes += [c - 4.0 for c in closes[-5:]]
    closes.append(closes[-1] + 8.0)
    for i, c in enumerate(closes[:-1]):
        strat.on_bar(_bar(i, close=c, volume=1_000_000))
    strat.on_bar(_bar(len(closes) - 1, close=closes[-1], volume=2_500_000))
    # Feed 15 more bars on a sharp downtrend to flip the EMA stack.
    last_close = closes[-1]
    for i in range(15):
        last_close -= 0.5
        strat.on_bar(_bar(len(closes) + i, close=last_close, volume=1_000_000))
    # Strategy should have flattened by now.
    assert strat._last_direction is Direction.FLAT


def test_indicator_helpers_exposed() -> None:
    """Cheap sanity check that the new indicator functions return values
    on a well-formed series."""
    from aitrade.strategy.indicators import (
        rolling_vwap,
        volume_ma,
        volume_ratio,
        vwap_distance_pct,
    )
    bars = [_bar(i, close=100.0 + i * 0.1) for i in range(25)]
    v = rolling_vwap(bars, 20)
    vma = volume_ma(bars, 20)
    vr = volume_ratio(bars, 20)
    vd = vwap_distance_pct(bars, 20)
    assert v is not None
    assert v > 0
    assert vma is not None
    assert vma == 1_000_000.0
    assert vr is not None
    assert 0.9 < vr < 1.1
    assert vd is not None
    assert -10 < vd < 10
