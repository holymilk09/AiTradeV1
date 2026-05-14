"""Top-movers universe layer — Alpaca's own most-actives + market-movers feed.

This is the *deterministic*, *fast*, *free* alternative to web-search-driven
discovery. Alpaca exposes two screener endpoints we tap in parallel:

  1. ``most_actives``   — top symbols by cumulative session volume / trade count
  2. ``market_movers``  — top gainers + top losers by % change for the day

Both feed straight off the broker's own market data — no LLM, no rate-limit
risk, no VPN issues, no sentiment overfit. Movers are scored by abs(% change)
× log(volume); most-actives get a baseline based on volume rank.

If Alpaca's screener errors (auth issue, API outage), the finder returns ``[]``
gracefully. Discovery is best-effort, never load-bearing.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.discovery.scorer import DiscoveredTicker

if TYPE_CHECKING:
    from alpaca.data.historical.screener import ScreenerClient


class MoversFinder:
    """Pull top market movers + most-actives from Alpaca's screener API.

    The output mirrors :class:`aitrade.discovery.scorer.DiscoveredTicker` so it
    composes with the watchlist and the optional web-discovery agent through
    the same engine plumbing.
    """

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: ScreenerClient | None = None  # lazy

    def _get_client(self) -> ScreenerClient | None:
        """Construct (and cache) the alpaca-py ScreenerClient, or None if no creds."""
        if self._client is not None:
            return self._client
        if not self._settings.has_credentials:
            logger.warning("MoversFinder: Alpaca credentials missing; returning no movers")
            return None
        from alpaca.data.historical.screener import ScreenerClient

        self._client = ScreenerClient(
            api_key=self._settings.alpaca_api_key.get_secret_value(),
            secret_key=self._settings.alpaca_secret_key.get_secret_value(),
        )
        return self._client

    def find(self, *, top_movers: int = 10, top_actives: int = 10) -> list[DiscoveredTicker]:
        """Return movers + most-actives unioned and scored. Never raises.

        Each ticker gets one ``DiscoveredTicker`` row. If a symbol shows up in
        both feeds, the higher score wins; the snippets accumulate.
        """
        client = self._get_client()
        if client is None:
            return []

        from alpaca.data.enums import MarketType, MostActivesBy
        from alpaca.data.requests import MarketMoversRequest, MostActivesRequest

        now = datetime.now(UTC)
        seen: dict[str, DiscoveredTicker] = {}

        # 1. Top gainers + losers
        try:
            movers = client.get_market_movers(
                MarketMoversRequest(market_type=MarketType.STOCKS, top=top_movers),
            )
        except Exception as e:
            logger.warning("MoversFinder: get_market_movers failed: {} ({})", type(e).__name__, e)
            movers = None

        if movers is not None:
            for direction, items in (("gainer", getattr(movers, "gainers", [])),
                                      ("loser", getattr(movers, "losers", []))):
                for m in items:
                    sym = str(m.symbol).upper()
                    pct = float(m.percent_change)
                    score = min(1.0, abs(pct) / 10.0)  # 10% move ⇒ score 1.0
                    snippet = f"{direction} {pct:+.2f}% @ ${m.price:.2f}"
                    seen[sym] = DiscoveredTicker(
                        symbol=sym,
                        mention_count=1,
                        source_weight=1.0,
                        recency_minutes=0.0,
                        buzz_score=score,
                        discovered_at=now,
                        evidence=[snippet],
                    )

        # 2. Most actives (volume-ranked)
        try:
            actives = client.get_most_actives(
                MostActivesRequest(by=MostActivesBy.VOLUME, top=top_actives),
            )
        except Exception as e:
            logger.warning("MoversFinder: get_most_actives failed: {} ({})", type(e).__name__, e)
            actives = None

        if actives is not None:
            entries = list(getattr(actives, "most_actives", []))
            n = max(1, len(entries))
            for rank, a in enumerate(entries):
                sym = str(a.symbol).upper()
                # Higher rank ⇒ higher score; saturate by volume magnitude.
                rank_score = (n - rank) / n
                vol_score = min(1.0, math.log(1 + float(a.volume)) / 20.0)
                score = max(rank_score, vol_score)
                snippet = f"most-active rank {rank + 1} vol={int(a.volume):,}"
                existing = seen.get(sym)
                if existing is None or score > existing.buzz_score:
                    seen[sym] = DiscoveredTicker(
                        symbol=sym,
                        mention_count=existing.mention_count + 1 if existing else 1,
                        source_weight=1.0,
                        recency_minutes=0.0,
                        buzz_score=score,
                        discovered_at=now,
                        evidence=[*(existing.evidence if existing else []), snippet],
                    )
                elif existing is not None:
                    existing.evidence.append(snippet)

        out = list(seen.values())
        out.sort(key=lambda d: d.buzz_score, reverse=True)
        return out
