"""Pattern detector unit tests — synthetic bars, no network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.patterns import (
    BreakoutDetector,
    GapAndGoDetector,
    PatternDetector,
    PatternSignal,
    PullbackDetector,
    UnusualVolumeDetector,
    VolumePriceTrendDetector,
    get_all_detectors,
)
from aitrade.strategy.signal import Direction


def _bars(
    closes: list[float],
    volumes: list[float] | None = None,
    *,
    symbol: str = "T",
    start: datetime | None = None,
    step: timedelta = timedelta(days=1),
) -> list[Bar]:
    """Build a synthetic bar series. Volumes default to flat 1000."""
    if volumes is None:
        volumes = [1000.0] * len(closes)
    if len(volumes) != len(closes):
        raise ValueError("closes and volumes length mismatch")
    base = start if start is not None else datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol=symbol,
            timestamp=base + step * i,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=v,
        )
        for i, (c, v) in enumerate(zip(closes, volumes, strict=True))
    ]


# --------------------------------------------------------------------------- #
# Registry / protocol sanity
# --------------------------------------------------------------------------- #


def test_registry_returns_all_five_detectors() -> None:
    detectors = get_all_detectors()
    assert len(detectors) == 5
    for d in detectors:
        assert isinstance(d, PatternDetector)
        assert isinstance(d.name, str)
        assert d.name


# --------------------------------------------------------------------------- #
# VolumePriceTrendDetector
# --------------------------------------------------------------------------- #


def test_volume_price_trend_fires_on_three_up_days_with_heavy_volume() -> None:
    # 25 flat days at vol 1000, then 3 strictly up days with volume = 2000 (2x baseline).
    closes = [100.0] * 25 + [101.0, 102.0, 103.0]
    volumes = [1000.0] * 25 + [2000.0, 2000.0, 2000.0]
    sig = VolumePriceTrendDetector().detect(_bars(closes, volumes))
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert 0.0 < sig.score <= 1.0
    assert {"days_up", "avg_vol_ratio", "pct_change"} <= set(sig.evidence)
    assert sig.evidence["days_up"] == 3.0
    assert sig.evidence["avg_vol_ratio"] > 1.15


def test_volume_price_trend_no_fire_on_low_volume() -> None:
    closes = [100.0] * 25 + [101.0, 102.0, 103.0]
    volumes = [1000.0] * 28  # no volume expansion
    assert VolumePriceTrendDetector().detect(_bars(closes, volumes)) is None


def test_volume_price_trend_no_fire_when_a_day_is_down() -> None:
    closes = [100.0] * 25 + [101.0, 100.5, 103.0]  # middle day down
    volumes = [1000.0] * 25 + [2000.0, 2000.0, 2000.0]
    assert VolumePriceTrendDetector().detect(_bars(closes, volumes)) is None


# --------------------------------------------------------------------------- #
# BreakoutDetector
# --------------------------------------------------------------------------- #


def test_breakout_fires_on_new_high_with_volume() -> None:
    # 25 flat closes at 100, then a breakout to 105 on 3x volume.
    closes = [100.0] * 25 + [105.0]
    volumes = [1000.0] * 25 + [3000.0]
    sig = BreakoutDetector().detect(_bars(closes, volumes))
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert 0.0 < sig.score <= 1.0
    assert sig.evidence["prior_high"] == 100.0
    assert sig.evidence["breakout_pct"] > 0.0
    assert sig.evidence["vol_ratio"] > 1.5


def test_breakout_no_fire_without_volume_kicker() -> None:
    closes = [100.0] * 25 + [105.0]
    volumes = [1000.0] * 26  # flat volume — no confirmation
    assert BreakoutDetector().detect(_bars(closes, volumes)) is None


def test_breakout_no_fire_when_close_below_prior_high() -> None:
    closes = [100.0] * 24 + [110.0, 105.0]  # prior high 110, latest 105
    volumes = [1000.0] * 25 + [3000.0]
    assert BreakoutDetector().detect(_bars(closes, volumes)) is None


# --------------------------------------------------------------------------- #
# UnusualVolumeDetector
# --------------------------------------------------------------------------- #


def test_unusual_volume_fires_long_on_up_close() -> None:
    # Prior window needs non-zero std for the z-score to be defined.
    base_vols = [1000.0 + (i % 2) * 50.0 for i in range(20)]
    closes = [100.0] * 20 + [101.0]
    volumes = [*base_vols, 10_000.0]
    sig = UnusualVolumeDetector().detect(_bars(closes, volumes))
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert 0.0 < sig.score <= 1.0
    assert sig.evidence["z_score"] > 2.0
    assert sig.evidence["vol_mean"] > 0
    assert sig.evidence["vol_std"] > 0


def test_unusual_volume_emits_flat_on_down_close() -> None:
    # Add some std to the prior window so std > 0.
    base_vols = [1000.0 + (i % 2) * 50.0 for i in range(20)]
    closes = [100.0] * 20 + [99.0]  # latest close down
    volumes = [*base_vols, 10_000.0]
    sig = UnusualVolumeDetector().detect(_bars(closes, volumes))
    assert sig is not None
    assert sig.direction == Direction.FLAT


def test_unusual_volume_no_fire_on_normal_volume() -> None:
    # Wide spread in the prior window → latest 1050 sits well within 2 std.
    base_vols = [1000.0 + (i % 4) * 250.0 for i in range(20)]
    closes = [100.0] * 20 + [101.0]
    volumes = [*base_vols, 1050.0]
    assert UnusualVolumeDetector().detect(_bars(closes, volumes)) is None


# --------------------------------------------------------------------------- #
# PullbackDetector
# --------------------------------------------------------------------------- #


def test_pullback_fires_in_uptrend_at_sma50_with_cool_rsi() -> None:
    # Build 199 strongly rising closes (so SMA50 > SMA200), then a final
    # bar that drops back near the 50-SMA to cool RSI into [30, 45].
    rising = [50.0 + i * 0.5 for i in range(199)]
    bars = _bars(rising)
    closes = [b.close for b in bars]
    from aitrade.strategy.indicators import sma as _sma

    sma50 = _sma(closes, 50)
    assert sma50 is not None
    # Drop the final bar to right at SMA50 — that's the pullback.
    final_close = sma50
    bars.append(
        Bar(
            symbol="T",
            timestamp=bars[-1].timestamp + timedelta(days=1),
            open=final_close,
            high=final_close + 0.5,
            low=final_close - 0.5,
            close=final_close,
            volume=1000.0,
        )
    )
    sig = PullbackDetector().detect(bars)
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert 0.0 <= sig.score <= 1.0
    assert {"rsi_14", "dist_to_sma50_pct", "sma50", "sma200"} <= set(sig.evidence)
    assert 30.0 <= sig.evidence["rsi_14"] <= 45.0


def test_pullback_no_fire_on_strong_uptrend_with_hot_rsi() -> None:
    # Strictly rising series → RSI ~100, far from the [30, 45] band.
    rising = [50.0 + i * 0.5 for i in range(220)]
    assert PullbackDetector().detect(_bars(rising)) is None


def test_pullback_no_fire_with_insufficient_history() -> None:
    closes = [100.0 + i * 0.1 for i in range(199)]  # one short of the 200 minimum
    assert PullbackDetector().detect(_bars(closes)) is None


# --------------------------------------------------------------------------- #
# GapAndGoDetector
# --------------------------------------------------------------------------- #


def _intraday_gap_bars(*, gap_open: float, holds_above: bool) -> list[Bar]:
    """Build day-1 close + day-2 5-min bars for the gap-and-go scenarios."""
    day1 = datetime(2024, 1, 1, 20, 55, tzinfo=UTC)  # last 5-min bar of "day 1"
    day2_open = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)  # first bar of "day 2"
    bars: list[Bar] = [
        Bar(
            symbol="T",
            timestamp=day1,
            open=100.0,
            high=100.5,
            low=99.5,
            close=100.0,
            volume=1000.0,
        ),
        Bar(
            symbol="T",
            timestamp=day2_open,
            open=gap_open,
            high=gap_open + 0.5,
            low=gap_open - 0.5,
            close=gap_open + 0.2,
            volume=2000.0,
        ),
    ]
    held_close = gap_open + 0.1 if holds_above else gap_open - 1.0
    for i in range(1, 4):  # three follow-up 5-min bars covering the 15-min hold
        bars.append(
            Bar(
                symbol="T",
                timestamp=day2_open + timedelta(minutes=5 * i),
                open=gap_open,
                high=gap_open + 0.5,
                low=gap_open - 0.5,
                close=held_close,
                volume=1500.0,
            )
        )
    return bars


def test_gap_and_go_fires_when_open_holds() -> None:
    bars = _intraday_gap_bars(gap_open=103.0, holds_above=True)
    sig = GapAndGoDetector().detect(bars)
    assert sig is not None
    assert sig.direction == Direction.LONG
    assert 0.0 < sig.score <= 1.0
    assert {"gap_pct", "held_mins", "session_open"} <= set(sig.evidence)
    assert sig.evidence["session_open"] == 103.0
    assert sig.evidence["gap_pct"] > 0.02


def test_gap_and_go_no_fire_when_price_breaks_back_below_open() -> None:
    bars = _intraday_gap_bars(gap_open=103.0, holds_above=False)
    assert GapAndGoDetector().detect(bars) is None


def test_gap_and_go_no_fire_without_sufficient_gap() -> None:
    # 1.005x gap is below the default 1.02 threshold.
    bars = _intraday_gap_bars(gap_open=100.5, holds_above=True)
    assert GapAndGoDetector().detect(bars) is None


# --------------------------------------------------------------------------- #
# Cross-cutting invariants
# --------------------------------------------------------------------------- #


def test_all_detectors_return_none_on_empty_bars() -> None:
    for d in get_all_detectors():
        assert d.detect([]) is None


def test_pattern_signal_is_frozen() -> None:
    sig = PatternSignal(
        name="x",
        symbol="T",
        score=0.5,
        direction=Direction.LONG,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        evidence={"a": 1.0},
    )
    try:
        sig.score = 0.9  # type: ignore[misc]
    except Exception:  # noqa: BLE001 — frozen dataclass raises FrozenInstanceError
        return
    raise AssertionError("PatternSignal should be frozen")
