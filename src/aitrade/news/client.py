"""Alpaca News client — recent headlines per symbol with TTL caching.

The wrapper is defensive by design: any failure (network, auth, SDK) is logged
and converted to ``{symbol: []}`` so callers (the LLM prompt builder) never have
to defend themselves against exceptions originating here. News is *flavor* on
top of indicators, not a hard dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import cast

from loguru import logger

from aitrade.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One news headline projected for our use.

    ``age_minutes`` is computed at fetch time relative to "now"; it's stored on
    the dataclass so the LLM-facing dict in :meth:`to_compact` doesn't need to
    know the wall clock.
    """

    id: int
    headline: str
    source: str
    url: str | None
    summary: str
    symbols: list[str]
    created_at: datetime
    age_minutes: float

    def to_compact(self) -> dict[str, str]:
        """Return a small dict for stuffing into an LLM prompt.

        Truncates ``summary`` to ~200 chars to keep prompt budget bounded.
        """
        summary = self.summary or ""
        if len(summary) > 200:
            summary = summary[:200].rstrip() + "…"
        return {
            "headline": self.headline,
            "source": self.source,
            "age_min": f"{self.age_minutes:.0f}",
            "summary": summary,
        }


# Cache key: (frozenset of symbols, lookback_hours, limit_per_symbol).
_CacheKey = tuple[frozenset[str], int, int]


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float  # monotonic seconds
    payload: dict[str, list[NewsItem]] = field(default_factory=dict)


class AlpacaNewsClient:
    """Thin wrapper over ``alpaca.data.historical.news.NewsClient``.

    Caches per ``(symbols, lookback, limit)`` for ``ttl_secs`` so repeated
    prompts within a tick don't burn rate limit. ``ttl_secs=0`` disables caching
    (every call hits the SDK).

    The SDK client is created lazily on first use and exposed as ``self._client``
    so tests can inject a stub directly (mirrors ``ClaudeReasoner``).
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        ttl_secs: int = 300,
    ) -> None:
        self._settings = settings or get_settings()
        self._ttl_secs = ttl_secs
        self._client: object | None = None  # lazy; tests may set directly
        self._cache: dict[_CacheKey, _CacheEntry] = {}

    def _get_client(self) -> object:
        if self._client is None:
            from alpaca.data.historical.news import NewsClient

            self._client = NewsClient(
                api_key=self._settings.alpaca_api_key.get_secret_value(),
                secret_key=self._settings.alpaca_secret_key.get_secret_value(),
            )
        return self._client

    def fetch(
        self,
        symbols: list[str],
        *,
        lookback_hours: int = 24,
        limit_per_symbol: int = 5,
    ) -> dict[str, list[NewsItem]]:
        """Fetch recent news per symbol.

        Returns a dict keyed by *every* requested symbol; symbols with no
        coverage map to an empty list. Never raises — failures are logged and
        produce empty lists.
        """
        if not symbols:
            return {}

        # Preserve insertion order for the result; dedupe defensively.
        seen: set[str] = set()
        ordered: list[str] = []
        for sym in symbols:
            if sym not in seen:
                seen.add(sym)
                ordered.append(sym)

        cache_key: _CacheKey = (frozenset(ordered), lookback_hours, limit_per_symbol)
        now = monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and cached.expires_at > now:
            return {sym: list(cached.payload.get(sym, [])) for sym in ordered}

        try:
            articles = self._fetch_articles(
                ordered,
                lookback_hours=lookback_hours,
                # Pull a bit more than strictly necessary so the per-symbol
                # partition still has enough after multi-symbol overlap.
                fetch_limit=max(limit_per_symbol * len(ordered) * 2, 10),
            )
        except Exception as exc:  # noqa: BLE001 — defensive boundary
            logger.warning("news fetch failed for {}: {}", ordered, exc)
            empty: dict[str, list[NewsItem]] = {sym: [] for sym in ordered}
            # Don't cache failures — let the next call retry.
            return empty

        partitioned = self._partition(articles, ordered, limit_per_symbol)

        if self._ttl_secs > 0:
            self._cache[cache_key] = _CacheEntry(
                expires_at=now + self._ttl_secs,
                payload={sym: list(items) for sym, items in partitioned.items()},
            )

        return partitioned

    def _fetch_articles(
        self,
        symbols: list[str],
        *,
        lookback_hours: int,
        fetch_limit: int,
    ) -> list[NewsItem]:
        from alpaca.data.requests import NewsRequest

        end = datetime.now(UTC)
        start = end - timedelta(hours=lookback_hours)

        req = NewsRequest(
            start=start,
            end=end,
            symbols=",".join(symbols),
            limit=fetch_limit,
            sort="desc",  # newest first
            include_content=False,
            exclude_contentless=True,
        )

        client = self._get_client()
        # The SDK's get_news returns NewsSet (or RawData if use_raw_data=True).
        # We don't construct with raw_data, so it's NewsSet.
        result = client.get_news(req)  # type: ignore[attr-defined]
        raw_articles = self._extract_articles(result)
        return [_to_news_item(article, end) for article in raw_articles]

    @staticmethod
    def _extract_articles(result: object) -> list[object]:
        """Pull the list of News objects out of NewsSet (or compatible stub).

        NewsSet stores them at ``data["news"]``; some stubs may expose a plain
        ``.news`` attribute. Support both.
        """
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            articles = data.get("news", [])
            return list(cast("list[object]", articles))
        news = getattr(result, "news", None)
        if news is not None:
            return list(cast("list[object]", news))
        return []

    @staticmethod
    def _partition(
        articles: list[NewsItem],
        symbols: list[str],
        limit_per_symbol: int,
    ) -> dict[str, list[NewsItem]]:
        """Bucket articles by membership in each symbol's ``symbols`` list.

        Articles are assumed already sorted newest-first; we keep that ordering
        per bucket and trim to ``limit_per_symbol``.
        """
        out: dict[str, list[NewsItem]] = {sym: [] for sym in symbols}
        for art in articles:
            for sym in symbols:
                if len(out[sym]) >= limit_per_symbol:
                    continue
                if sym in art.symbols:
                    out[sym].append(art)
        # Ensure newest-first within each bucket (input is already desc, but
        # guard against stubs that don't sort).
        for sym, items in out.items():
            items.sort(key=lambda n: n.created_at, reverse=True)
            out[sym] = items[:limit_per_symbol]
        return out


def _to_news_item(article: object, now: datetime) -> NewsItem:
    """Coerce an SDK News (or compatible duck-typed object) into a NewsItem."""
    created_at = cast(datetime, article.created_at)  # type: ignore[attr-defined]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    age = (now - created_at).total_seconds() / 60.0
    symbols_raw = getattr(article, "symbols", []) or []
    return NewsItem(
        id=int(article.id),  # type: ignore[attr-defined]
        headline=str(getattr(article, "headline", "")),
        source=str(getattr(article, "source", "")),
        url=cast("str | None", getattr(article, "url", None)),
        summary=str(getattr(article, "summary", "") or ""),
        symbols=[str(s) for s in symbols_raw],
        created_at=created_at,
        age_minutes=max(age, 0.0),
    )
