"""Bar-freshness guards.

Callers feed live signals through :func:`assert_bars_fresh` to refuse to
trade on stale data — e.g. when the websocket has fallen behind, the
historical fetch returned a closed-market gap, or a cache lookup served a
multi-hour-old window.

Use it as a precondition; never silently fall back to a stale snapshot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar, Timeframe

_THRESHOLDS: dict[Timeframe, timedelta] = {
    Timeframe.MIN_1: timedelta(minutes=3),
    Timeframe.MIN_5: timedelta(minutes=10),
    Timeframe.MIN_15: timedelta(minutes=25),
    Timeframe.HOUR_1: timedelta(hours=1, minutes=10),
    Timeframe.DAY_1: timedelta(days=2),
}


class StaleDataError(RuntimeError):
    """Raised when bar history is older than the freshness budget.

    Callers should treat this as a hard stop: refuse to size or place orders
    until fresh data is available.
    """


def assert_bars_fresh(
    bars: list[Bar],
    timeframe: Timeframe,
    *,
    now: datetime | None = None,
) -> None:
    """Raise :class:`StaleDataError` if the bar window is unusable.

    Parameters
    ----------
    bars:
        Recent bars, oldest-first. An empty list is always stale.
    timeframe:
        Resolution of the bars; selects the per-timeframe staleness budget.
    now:
        Override the wall clock — used by tests. Defaults to ``datetime.now(UTC)``.
    """

    if not bars:
        raise StaleDataError(f"no bars supplied for timeframe={timeframe.value}")

    threshold = _THRESHOLDS[timeframe]
    current = now if now is not None else datetime.now(UTC)
    last = bars[-1].timestamp
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)

    age = current - last
    if age > threshold:
        raise StaleDataError(
            f"last bar for timeframe={timeframe.value} is {age} old "
            f"(threshold={threshold})"
        )
