"""News client behavior — partitioning, limits, TTL cache, failure paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.news import AlpacaNewsClient, NewsItem


@dataclass
class _StubNews:
    """Duck-typed stand-in for ``alpaca.data.models.news.News``."""

    id: int
    headline: str
    source: str
    url: str | None
    summary: str
    symbols: list[str]
    created_at: datetime


@dataclass
class _StubNewsSet:
    """Duck-typed stand-in for ``alpaca.data.models.news.NewsSet``."""

    articles: list[_StubNews]

    @property
    def data(self) -> dict[str, list[_StubNews]]:
        return {"news": self.articles}


@dataclass
class _StubClient:
    """Captures get_news calls and returns a canned NewsSet."""

    response: _StubNewsSet | None = None
    raise_with: Exception | None = None
    calls: int = 0
    last_request: object | None = field(default=None)

    def get_news(self, req: object) -> _StubNewsSet:
        self.calls += 1
        self.last_request = req
        if self.raise_with is not None:
            raise self.raise_with
        assert self.response is not None
        return self.response


def _settings() -> Settings:
    return Settings(
        alpaca_api_key=SecretStr("k"),
        alpaca_secret_key=SecretStr("s"),
    )


def _ts(minutes_ago: float) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes_ago)


def _article(
    aid: int,
    *,
    symbols: list[str],
    minutes_ago: float = 1.0,
    headline: str | None = None,
    summary: str = "summary",
) -> _StubNews:
    return _StubNews(
        id=aid,
        headline=headline or f"news-{aid}",
        source="Benzinga",
        url=f"https://example.com/{aid}",
        summary=summary,
        symbols=symbols,
        created_at=_ts(minutes_ago),
    )


def _make_client(stub: _StubClient, *, ttl_secs: int = 300) -> AlpacaNewsClient:
    c = AlpacaNewsClient(settings=_settings(), ttl_secs=ttl_secs)
    c._client = stub  # inject stub directly — same pattern as ClaudeReasoner
    return c


def test_fetch_partitions_by_symbol() -> None:
    articles = [
        _article(1, symbols=["AAPL"], minutes_ago=1),
        _article(2, symbols=["MSFT"], minutes_ago=2),
        _article(3, symbols=["AAPL", "MSFT"], minutes_ago=3),
        _article(4, symbols=["NVDA"], minutes_ago=4),  # not requested
        _article(5, symbols=["AAPL"], minutes_ago=5),
    ]
    stub = _StubClient(response=_StubNewsSet(articles=articles))
    client = _make_client(stub)

    out = client.fetch(["AAPL", "MSFT", "GOOG"])

    assert set(out.keys()) == {"AAPL", "MSFT", "GOOG"}
    assert [n.id for n in out["AAPL"]] == [1, 3, 5]
    assert [n.id for n in out["MSFT"]] == [2, 3]
    assert out["GOOG"] == []  # missing symbol — empty list, not absent


def test_fetch_respects_limit_per_symbol() -> None:
    articles = [
        _article(i, symbols=["AAPL"], minutes_ago=float(i)) for i in range(1, 11)
    ]
    stub = _StubClient(response=_StubNewsSet(articles=articles))
    client = _make_client(stub)

    out = client.fetch(["AAPL"], limit_per_symbol=3)

    assert len(out["AAPL"]) == 3
    # Newest first — minutes_ago=1, 2, 3 → ids 1, 2, 3.
    assert [n.id for n in out["AAPL"]] == [1, 2, 3]


def test_fetch_caches_within_ttl() -> None:
    articles = [_article(1, symbols=["AAPL"])]
    stub = _StubClient(response=_StubNewsSet(articles=articles))
    client = _make_client(stub, ttl_secs=300)

    first = client.fetch(["AAPL"])
    second = client.fetch(["AAPL"])

    assert stub.calls == 1
    assert [n.id for n in first["AAPL"]] == [1]
    assert [n.id for n in second["AAPL"]] == [1]


def test_fetch_cache_expires() -> None:
    articles = [_article(1, symbols=["AAPL"])]
    stub = _StubClient(response=_StubNewsSet(articles=articles))
    client = _make_client(stub, ttl_secs=0)  # disabled

    client.fetch(["AAPL"])
    client.fetch(["AAPL"])

    assert stub.calls == 2


def test_fetch_returns_empty_on_failure() -> None:
    stub = _StubClient(raise_with=RuntimeError("boom"))
    client = _make_client(stub)

    out = client.fetch(["AAPL", "MSFT"])

    assert out == {"AAPL": [], "MSFT": []}
    # Failure path doesn't cache — next call retries.
    client.fetch(["AAPL", "MSFT"])
    assert stub.calls == 2


def test_fetch_empty_symbols_returns_empty_dict() -> None:
    stub = _StubClient(response=_StubNewsSet(articles=[]))
    client = _make_client(stub)
    assert client.fetch([]) == {}
    assert stub.calls == 0


def test_news_item_to_compact_truncates() -> None:
    long_summary = "x" * 500
    item = NewsItem(
        id=1,
        headline="big move",
        source="Reuters",
        url="https://example.com/1",
        summary=long_summary,
        symbols=["AAPL"],
        created_at=_ts(15.0),
        age_minutes=15.0,
    )

    compact = item.to_compact()

    assert compact["headline"] == "big move"
    assert compact["source"] == "Reuters"
    assert compact["age_min"] == "15"
    # Truncated to ~200 chars (with ellipsis we allow a tiny overshoot).
    assert len(compact["summary"]) <= 220
    assert len(compact["summary"]) > 100  # something non-trivial preserved


def test_news_item_to_compact_short_summary_unchanged() -> None:
    item = NewsItem(
        id=2,
        headline="h",
        source="s",
        url=None,
        summary="short",
        symbols=["AAPL"],
        created_at=_ts(1.0),
        age_minutes=1.0,
    )
    assert item.to_compact()["summary"] == "short"


@pytest.mark.parametrize("symbols", [["AAPL"], ["AAPL", "MSFT", "GOOG"]])
def test_fetch_keys_match_requested(symbols: list[str]) -> None:
    stub = _StubClient(response=_StubNewsSet(articles=[]))
    client = _make_client(stub)
    out = client.fetch(symbols)
    assert list(out.keys()) == symbols
