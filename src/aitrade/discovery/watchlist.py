"""Static configurable watchlist — the foundational universe layer.

Read from the ``AITRADE_WATCHLIST`` env var (comma-separated) or fall back
to a sensible large-cap + popular-name default. Always available; zero
network cost; deterministic.

This is the *primary* universe driver. Movers (price-based) and the optional
web-discovery agent compose **on top** of it — they don't replace it.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from aitrade.discovery.scorer import DiscoveredTicker

# Sensible default watchlist — broad ETFs + Mag 7 + top-volume names. Override
# via env var ``AITRADE_WATCHLIST=AAPL,MSFT,...``. Keep it short by default
# (≈30 symbols) so per-cycle bar fetches stay snappy.
DEFAULT_WATCHLIST: tuple[str, ...] = (
    # Index ETFs — macro context anchors
    "SPY", "QQQ", "IWM", "DIA",
    # Mag 7 + top mega-caps
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AVGO", "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "WMT",
    # High-beta / popular options names
    "AMD", "NFLX", "DIS", "BA", "COIN", "PLTR", "SHOP", "UBER", "SNOW",
)


def load_watchlist() -> list[str]:
    """Return the active watchlist, env-overridden or default. Uppercase + deduped."""
    raw = os.environ.get("AITRADE_WATCHLIST", "").strip()
    if not raw:
        symbols = list(DEFAULT_WATCHLIST)
    else:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    seen: dict[str, None] = {}
    for s in symbols:
        seen.setdefault(s, None)
    return list(seen)


def watchlist_as_discovered(symbols: list[str] | None = None) -> list[DiscoveredTicker]:
    """Project the watchlist into ``DiscoveredTicker`` shape so the engine's
    universe-merge logic can treat it like any other discovery source.

    Watchlist tickers carry a baseline buzz score of 1.0 — enough to keep them
    in candidate ranking even on quiet days, but movers / web buzz can still
    rank higher when something's actually happening.
    """
    syms = symbols if symbols is not None else load_watchlist()
    now = datetime.now(UTC)
    return [
        DiscoveredTicker(
            symbol=s,
            mention_count=1,
            source_weight=1.0,  # treat watchlist as tier-1 by default
            recency_minutes=0.0,
            buzz_score=1.0,
            discovered_at=now,
            evidence=["watchlist"],
        )
        for s in syms
    ]
