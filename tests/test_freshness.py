"""Bar freshness guards — empty lists and per-timeframe thresholds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar, Timeframe
from aitrade.market.freshness import StaleDataError, assert_bars_fresh


def _bar(ts: datetime) -> Bar:
    return Bar(
        symbol="T",
        timestamp=ts,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        volume=1.0,
    )


def test_empty_list_raises() -> None:
    with pytest.raises(StaleDataError):
        assert_bars_fresh([], Timeframe.MIN_5)


@pytest.mark.parametrize(
    ("timeframe", "stale_age"),
    [
        (Timeframe.MIN_1, timedelta(minutes=10)),
        (Timeframe.MIN_5, timedelta(minutes=30)),
        (Timeframe.MIN_15, timedelta(hours=1)),
        (Timeframe.HOUR_1, timedelta(hours=4)),
        (Timeframe.DAY_1, timedelta(days=5)),
    ],
)
def test_raises_on_old_bars(timeframe: Timeframe, stale_age: timedelta) -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)
    bars = [_bar(now - stale_age)]
    with pytest.raises(StaleDataError):
        assert_bars_fresh(bars, timeframe, now=now)


@pytest.mark.parametrize(
    ("timeframe", "fresh_age"),
    [
        (Timeframe.MIN_1, timedelta(minutes=1)),
        (Timeframe.MIN_5, timedelta(minutes=4)),
        (Timeframe.MIN_15, timedelta(minutes=10)),
        (Timeframe.HOUR_1, timedelta(minutes=30)),
        (Timeframe.DAY_1, timedelta(hours=12)),
    ],
)
def test_does_not_raise_on_fresh_bars(
    timeframe: Timeframe, fresh_age: timedelta
) -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)
    bars = [_bar(now - fresh_age)]
    assert_bars_fresh(bars, timeframe, now=now)


def test_naive_timestamps_are_treated_as_utc() -> None:
    now = datetime(2026, 4, 24, 15, 0, tzinfo=UTC)
    naive = (now - timedelta(minutes=1)).replace(tzinfo=None)
    bars = [_bar(naive)]
    assert_bars_fresh(bars, Timeframe.MIN_5, now=now)
