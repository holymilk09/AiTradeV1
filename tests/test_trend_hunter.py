"""TrendHunter — deterministic multi-criteria trend scorer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.patterns.trend_hunter import TrendHunter, TrendScore
from aitrade.strategy.signal import Direction


def _bars(closes: list[float], *, base_volume: float = 1_000_000.0) -> list[Bar]:
    """Build OHLCV bars from a close-price series; H/L tracked tightly."""
    out: list[Bar] = []
    t = datetime(2026, 1, 1, tzinfo=UTC)
    prev = closes[0]
    for c in closes:
        hi = max(prev, c) * 1.01
        lo = min(prev, c) * 0.99
        out.append(
            Bar(
                symbol="TEST",
                timestamp=t,
                open=prev,
                high=hi,
                low=lo,
                close=c,
                volume=base_volume,
            )
        )
        prev = c
        t += timedelta(days=1)
    return out


def _uptrend(n: int = 220) -> list[Bar]:
    """Realistic uptrend — drift up with mild oscillation so RSI lands in the
    trend zone (40-70) instead of saturating at 100 on a perfectly linear series.
    """
    closes: list[float] = []
    for i in range(n):
        # Drift + sinusoidal noise. Period of 7 keeps RSI in [50, 70].
        import math

        closes.append(100.0 + i * 0.5 + 1.5 * math.sin(i / 7.0))
    bars = _bars(closes)
    last = bars[-1]
    bars[-1] = Bar(
        symbol=last.symbol,
        timestamp=last.timestamp,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume * 2.0,
    )
    return bars


def _downtrend(n: int = 220) -> list[Bar]:
    import math

    closes = [200.0 - i * 0.5 + 1.5 * math.sin(i / 7.0) for i in range(n)]
    return _bars(closes)


def _flat(n: int = 220) -> list[Bar]:
    """Choppy sideways market — significant up-down swings with no net drift."""
    import math

    closes = [100.0 + 5.0 * math.sin(i / 4.0) for i in range(n)]
    return _bars(closes)


def test_clean_uptrend_scores_high_and_long() -> None:
    th = TrendHunter()
    score = th.compute(_uptrend())
    assert score.score >= 0.7
    assert score.is_strong is True
    assert score.direction is Direction.LONG


def test_clean_downtrend_scores_high_and_short() -> None:
    th = TrendHunter()
    score = th.compute(_downtrend())
    assert score.score >= 0.6
    assert score.direction is Direction.SHORT


def test_flat_chop_does_not_register_as_strong_trend() -> None:
    """Choppy sideways markets MAY score moderately (sub-segments look trendy),
    but they must never trip the ``is_strong`` flag the floor-trader keys on,
    and the direction must come back FLAT (no consensus across components)."""
    th = TrendHunter()
    score = th.compute(_flat())
    assert score.is_strong is False
    assert score.direction is Direction.FLAT


def test_returns_zero_when_insufficient_history() -> None:
    th = TrendHunter()
    score = th.compute(_uptrend(n=30))  # below min_bars=60
    assert score.score == 0.0
    assert score.direction is Direction.FLAT
    assert score.components == {}


def test_to_dict_round_trips_components() -> None:
    th = TrendHunter()
    score = th.compute(_uptrend())
    d = score.to_dict()
    assert "score" in d
    assert "direction" in d
    assert "components" in d
    assert "is_strong" in d
    assert d["is_strong"] is True


def test_components_break_down_score_pieces() -> None:
    """Every component must contribute a [0, 1] sub-score."""
    th = TrendHunter()
    score = th.compute(_uptrend())
    # All textbook criteria fire on a clean uptrend.
    assert "ma_stack" in score.components
    assert "higher_highs_lows" in score.components
    assert "volume_confirm" in score.components
    assert "rsi_zone" in score.components
    assert "macd_positive" in score.components
    for v in score.components.values():
        assert 0.0 <= v <= 1.0


def test_volume_drop_lowers_volume_component() -> None:
    """Same price action with 0.5x recent volume → volume_confirm < 1.0."""
    th = TrendHunter()
    bars = _uptrend()
    # Replace the last bar with much lower volume.
    last = bars[-1]
    bars[-1] = Bar(
        symbol=last.symbol,
        timestamp=last.timestamp,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume * 0.1,
    )
    score = th.compute(bars)
    assert score.components["volume_confirm"] < 1.0


def test_trend_score_dataclass_is_frozen() -> None:
    """TrendScore should be safe to reuse — frozen + slotted."""
    score = TrendScore(score=0.5, direction=Direction.LONG, components={"a": 0.5})
    assert score.is_strong is False  # 0.5 < 0.7
