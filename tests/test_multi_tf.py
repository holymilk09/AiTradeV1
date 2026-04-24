"""MultiTimeframeSnapshot — building per-TF stacks from synthetic bars."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar, Timeframe
from aitrade.strategy.indicators import (
    MultiTimeframeSnapshot,
    compute_multi_tf_snapshot,
)


def _bars(closes: list[float], step: timedelta) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Bar(
            symbol="T",
            timestamp=start + step * i,
            open=c,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]


def test_compute_multi_tf_builds_per_tf_stacks() -> None:
    daily_closes = [100.0 + i * 0.5 for i in range(40)]
    hourly_closes = [200.0 + i * 0.25 for i in range(40)]
    bars_by_tf = {
        Timeframe.DAY_1: _bars(daily_closes, timedelta(days=1)),
        Timeframe.HOUR_1: _bars(hourly_closes, timedelta(hours=1)),
    }

    snap = compute_multi_tf_snapshot(bars_by_tf)

    assert isinstance(snap, MultiTimeframeSnapshot)
    assert set(snap.per_tf.keys()) == {Timeframe.DAY_1, Timeframe.HOUR_1}
    assert snap.per_tf[Timeframe.DAY_1].price == daily_closes[-1]
    assert snap.per_tf[Timeframe.HOUR_1].price == hourly_closes[-1]
    # Both uptrends → bull stack True on both
    assert snap.per_tf[Timeframe.DAY_1].bull_trend is True
    assert snap.per_tf[Timeframe.HOUR_1].bull_trend is True


def test_tf_accessor_returns_correct_snapshot() -> None:
    bars_by_tf = {
        Timeframe.DAY_1: _bars([100.0 + i for i in range(30)], timedelta(days=1)),
        Timeframe.MIN_5: _bars([50.0 + i for i in range(30)], timedelta(minutes=5)),
    }
    snap = compute_multi_tf_snapshot(bars_by_tf)
    assert snap.tf(Timeframe.DAY_1).price == 129.0
    assert snap.tf(Timeframe.MIN_5).price == 79.0


def test_tf_accessor_raises_keyerror_for_missing_timeframe() -> None:
    snap = compute_multi_tf_snapshot(
        {Timeframe.DAY_1: _bars([1.0] * 5, timedelta(days=1))}
    )
    with pytest.raises(KeyError):
        snap.tf(Timeframe.MIN_1)


def test_empty_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty bar list"):
        compute_multi_tf_snapshot({Timeframe.DAY_1: []})


def test_as_dict_returns_json_friendly_payload() -> None:
    snap = compute_multi_tf_snapshot(
        {Timeframe.DAY_1: _bars([100.0 + i * 0.5 for i in range(40)], timedelta(days=1))}
    )
    dumped = snap.as_dict()
    assert "1Day" in dumped
    assert dumped["1Day"]["symbol"] == "T"
    assert dumped["1Day"]["price"] == 119.5
