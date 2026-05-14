"""Reddit/WSB discovery — extracts tickers from public hot.json listings."""

from __future__ import annotations

import time
from typing import Any

from aitrade.discovery.extractor import TickerExtractor
from aitrade.discovery.reddit import RedditDiscoveryClient
from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.discovery.universe import build_universe


class _StubResp:
    def __init__(self, *, status: int = 200, payload: object | None = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _StubHTTP:
    """Captures ``.get`` calls and returns scripted responses keyed by sub."""

    def __init__(self, responses_by_sub: dict[str, _StubResp]) -> None:
        self._by_sub = responses_by_sub
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str] | None = None) -> _StubResp:
        self.calls.append((url, dict(params or {})))
        # URL form: https://www.reddit.com/r/<sub>/hot.json
        sub = url.rstrip("/").split("/")[-2]
        return self._by_sub.get(sub, _StubResp(status=404, payload={}))


def _post(
    *,
    title: str,
    selftext: str = "",
    score: int = 0,
    sub: str = "wallstreetbets",
    age_secs: float = 600.0,
) -> dict[str, Any]:
    return {
        "title": title,
        "selftext": selftext,
        "score": score,
        "created_utc": time.time() - age_secs,
        "subreddit": sub,
        "permalink": f"/r/{sub}/comments/x/",
    }


def _listing(posts: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap rows in Reddit's listing envelope."""
    return {"data": {"children": [{"kind": "t3", "data": p} for p in posts]}}


def _extractor(symbols: set[str] | None = None) -> TickerExtractor:
    return TickerExtractor(valid_tickers=symbols or {"AAPL", "TSLA", "NVDA", "GME"})


def test_extracts_tickers_from_titles_and_selftext() -> None:
    http = _StubHTTP(
        {
            "wallstreetbets": _StubResp(
                payload=_listing(
                    [
                        _post(title="$AAPL is mooning today", score=1500),
                        _post(
                            title="Calls on TSLA",
                            selftext="$TSLA going parabolic",
                            score=420,
                        ),
                    ]
                )
            ),
            "stocks": _StubResp(
                payload=_listing([_post(title="$NVDA earnings preview", score=80)])
            ),
        }
    )
    client = RedditDiscoveryClient(
        extractor=_extractor(),
        subs=("wallstreetbets", "stocks"),
        http=http,
    )
    out = client.discover()
    syms = {t.symbol for t in out}
    assert {"AAPL", "TSLA", "NVDA"} <= syms
    # WSB high-upvote symbols should outrank the small-sub low-upvote NVDA post.
    aapl = next(t for t in out if t.symbol == "AAPL")
    nvda = next(t for t in out if t.symbol == "NVDA")
    assert aapl.buzz_score > nvda.buzz_score


def test_evidence_snippets_include_subreddit_and_upvotes() -> None:
    http = _StubHTTP(
        {
            "wallstreetbets": _StubResp(
                payload=_listing([_post(title="$GME squeeze part 2", score=9999)])
            )
        }
    )
    client = RedditDiscoveryClient(
        extractor=_extractor(),
        subs=("wallstreetbets",),
        http=http,
    )
    out = client.discover()
    gme = next(t for t in out if t.symbol == "GME")
    assert any("r/wallstreetbets" in e and "9999↑" in e for e in gme.evidence)


def test_http_failure_returns_empty_list_silently() -> None:
    http = _StubHTTP(
        {"wallstreetbets": _StubResp(status=429, payload={"error": "ratelimited"})}
    )
    client = RedditDiscoveryClient(
        extractor=_extractor(),
        subs=("wallstreetbets",),
        http=http,
    )
    assert client.discover() == []


def test_malformed_json_returns_empty() -> None:
    class _BadResp:
        status_code = 200

        def json(self) -> object:
            raise ValueError("not json")

    class _BadHTTP:
        def get(self, url: str, params: dict[str, str] | None = None) -> _BadResp:
            return _BadResp()

    client = RedditDiscoveryClient(
        extractor=_extractor(),
        subs=("wallstreetbets",),
        http=_BadHTTP(),
    )
    assert client.discover() == []


def test_recency_decay_favors_fresh_posts() -> None:
    http = _StubHTTP(
        {
            "wallstreetbets": _StubResp(
                payload=_listing(
                    [
                        _post(title="$AAPL", score=500, age_secs=10 * 60),  # 10 min old
                        _post(title="$TSLA", score=500, age_secs=240 * 60),  # 4h old
                    ]
                )
            )
        }
    )
    client = RedditDiscoveryClient(
        extractor=_extractor(),
        subs=("wallstreetbets",),
        http=http,
    )
    out = client.discover()
    aapl = next(t for t in out if t.symbol == "AAPL")
    tsla = next(t for t in out if t.symbol == "TSLA")
    assert aapl.buzz_score > tsla.buzz_score


def test_invalid_tickers_are_dropped_by_extractor() -> None:
    # XYZQ is not in the valid universe — should be filtered out.
    http = _StubHTTP(
        {
            "wallstreetbets": _StubResp(
                payload=_listing(
                    [
                        _post(title="$XYZQ to the moon", score=1000),
                        _post(title="$AAPL solid", score=100),
                    ]
                )
            )
        }
    )
    client = RedditDiscoveryClient(
        extractor=_extractor({"AAPL"}),
        subs=("wallstreetbets",),
        http=http,
    )
    out = client.discover()
    syms = {t.symbol for t in out}
    assert syms == {"AAPL"}


def test_universe_uses_reddit_when_opted_in() -> None:
    """build_universe(use_reddit=True, ...) merges reddit results in."""

    class _StubReddit:
        def discover(self) -> list[DiscoveredTicker]:
            from datetime import UTC, datetime

            return [
                DiscoveredTicker(
                    symbol="GME",
                    mention_count=5,
                    source_weight=0.4,
                    recency_minutes=10.0,
                    buzz_score=0.9,
                    discovered_at=datetime.now(UTC),
                    evidence=["r/wallstreetbets 5000↑ — squeeze"],
                )
            ]

    out = build_universe(
        reddit=_StubReddit(),  # type: ignore[arg-type]
        use_watchlist=False,
        use_movers=False,
        use_reddit=True,
        use_web=False,
    )
    assert any(d.symbol == "GME" for d in out)


def test_universe_skips_reddit_when_disabled() -> None:
    """build_universe(use_reddit=False) must NOT call .discover() on reddit."""

    class _ExplodingReddit:
        def discover(self) -> list[DiscoveredTicker]:
            raise AssertionError("reddit should not be called when use_reddit=False")

    out = build_universe(
        reddit=_ExplodingReddit(),  # type: ignore[arg-type]
        use_watchlist=True,
        use_movers=False,
        use_reddit=False,
        use_web=False,
    )
    assert out  # watchlist still present
