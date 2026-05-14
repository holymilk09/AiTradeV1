"""Market-wide context: snapshots, regime, freshness guards.

Public surface:

- :class:`MarketSnapshot` — frozen point-in-time read of SPY/QQQ/VIX.
- :class:`MarketSnapshotFetcher` — TTL-cached fetcher with stale fallback.
- :class:`Regime` — coarse 2x2 regime label.
- :class:`StaleDataError` — raised when bar history is too old to trade on.
- :func:`assert_bars_fresh` — precondition helper for callers.
"""

from __future__ import annotations

from aitrade.market.freshness import StaleDataError, assert_bars_fresh
from aitrade.market.regime import Regime, classify_regime
from aitrade.market.snapshot import MarketSnapshot, MarketSnapshotFetcher

__all__ = [
    "MarketSnapshot",
    "MarketSnapshotFetcher",
    "Regime",
    "StaleDataError",
    "assert_bars_fresh",
    "classify_regime",
]
