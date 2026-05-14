"""Reddit/WSB social discovery — surface tickers buzzy on retail-trader subs.

Uses Reddit's public ``.json`` endpoints — no OAuth, no API key, no PRAW. All
we need is a ``User-Agent`` header. The flow per cycle:

  1. ``GET https://www.reddit.com/r/<sub>/hot.json?limit=N`` for each configured
     subreddit (default: r/wallstreetbets, r/stocks, r/options).
  2. For each post, extract tickers from ``title + selftext`` via the existing
     :class:`aitrade.discovery.extractor.TickerExtractor` (cashtag + contextual,
     validated against the Alpaca tradable universe).
  3. Aggregate per-symbol: total upvote-weighted mention count, freshest post
     age, accumulated evidence snippets ("WSB: 1.2k upvotes — '...title...'").
  4. Score with :class:`BuzzScorer` using the ``"reddit"`` source key (weight
     0.4 — same prior the agent uses).

Failure is silent: HTTP errors, ratelimiting, JSON parse errors, or an empty
result all yield ``[]``. Discovery is never load-bearing, and Reddit can — and
will — go down or rate-limit at the worst possible time.

Why not PRAW? It requires app credentials and gives us nothing the public JSON
endpoint doesn't already provide for a read-only crawl. Less surface area, no
extra config, no auth tokens to rotate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

from aitrade.discovery.extractor import TickerExtractor
from aitrade.discovery.scorer import BuzzScorer, DiscoveredTicker

# Conservative defaults — three subs cover most US-equity retail buzz without
# burning rate limits. r/options is intentionally included: options buzz often
# precedes a sharp underlying move.
_DEFAULT_SUBS: tuple[str, ...] = ("wallstreetbets", "stocks", "options")
_DEFAULT_POSTS_PER_SUB: int = 25
_TIMEOUT_SECS: float = 8.0
# Reddit asks every client to identify itself; a generic UA is rate-limited
# heavily, sometimes blocked. Use a project-specific token.
_USER_AGENT = "aitrade-discovery/0.1 (+https://github.com/holymilk09/AiTradeV1)"


@dataclass(frozen=True, slots=True)
class _Post:
    """A trimmed Reddit post — only what the scorer needs."""

    title: str
    selftext: str
    score: int
    created_utc: float
    subreddit: str
    permalink: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.selftext}"


class RedditDiscoveryClient:
    """Fetch hot posts from configured subreddits and score per-symbol buzz.

    Parameters
    ----------
    extractor:
        Validates ticker candidates against the tradable universe so we don't
        surface delisted or never-listed symbols (Reddit titles are noisy).
    scorer:
        Composes mention count + source weight + recency into a single buzz
        score; uses the ``"reddit"`` source key (weight 0.4).
    subs:
        Subreddits to crawl. Default is r/wallstreetbets, r/stocks, r/options.
    posts_per_sub:
        Number of "hot" posts to pull per subreddit (Reddit caps at 100).
    http:
        Inject a stub for tests; a real ``httpx.Client`` is built lazily.
    """

    def __init__(
        self,
        *,
        extractor: TickerExtractor,
        scorer: BuzzScorer | None = None,
        subs: tuple[str, ...] = _DEFAULT_SUBS,
        posts_per_sub: int = _DEFAULT_POSTS_PER_SUB,
        http: object | None = None,
    ) -> None:
        self._extractor = extractor
        self._scorer = scorer or BuzzScorer()
        self._subs = subs
        self._limit = max(1, min(100, posts_per_sub))
        self._http: object | None = http

    def _get_http(self) -> object:
        if self._http is None:
            self._http = httpx.Client(
                timeout=_TIMEOUT_SECS,
                headers={"User-Agent": _USER_AGENT},
            )
        return self._http

    def _fetch_sub(self, sub: str) -> list[_Post]:
        """Pull `limit` hot posts from one subreddit. Empty list on any error."""
        url = f"https://www.reddit.com/r/{sub}/hot.json"
        params = {"limit": str(self._limit)}
        http = self._get_http()
        try:
            resp = http.get(url, params=params)  # type: ignore[attr-defined]
        except Exception as e:  # network/protocol error — drop this sub
            logger.warning("reddit r/{} fetch failed: {}", sub, e)
            return []
        status = getattr(resp, "status_code", 0)
        if status != 200:
            logger.warning("reddit r/{} returned status {}", sub, status)
            return []
        try:
            payload: Any = resp.json()
        except Exception as e:
            logger.warning("reddit r/{} json parse failed: {}", sub, e)
            return []
        return _parse_listing(payload, sub)

    def discover(self) -> list[DiscoveredTicker]:
        """Crawl all configured subs and return scored ``DiscoveredTicker`` rows.

        Per-symbol aggregation: ``mention_count`` is the number of distinct posts
        the ticker appeared in; the buzz score weights each mention by the
        post's upvote count via ``log(1 + sum_upvotes)``. Recency is the age of
        the freshest mentioning post.
        """
        all_posts: list[_Post] = []
        for sub in self._subs:
            all_posts.extend(self._fetch_sub(sub))
        if not all_posts:
            return []

        now = datetime.now(UTC)
        # Per-symbol accumulators.
        mention_counts: dict[str, int] = {}
        upvote_sums: dict[str, int] = {}
        freshest_age_min: dict[str, float] = {}
        evidence_by_symbol: dict[str, list[str]] = {}

        for post in all_posts:
            tickers = self._extractor.extract(post.text)
            if not tickers:
                continue
            age_min = max(0.0, (time.time() - post.created_utc) / 60.0)
            score = max(0, post.score)
            for sym, count in tickers:
                mention_counts[sym] = mention_counts.get(sym, 0) + count
                upvote_sums[sym] = upvote_sums.get(sym, 0) + score
                prev = freshest_age_min.get(sym)
                if prev is None or age_min < prev:
                    freshest_age_min[sym] = age_min
                # Trim title for the snippet — full titles can be very long.
                title_trimmed = post.title[:120].replace("\n", " ")
                snippet = f"r/{post.subreddit} {score}↑ — {title_trimmed}"
                evidence_by_symbol.setdefault(sym, []).append(snippet)

        out: list[DiscoveredTicker] = []
        for sym, mentions in mention_counts.items():
            upvotes = upvote_sums.get(sym, 0)
            age_min = freshest_age_min.get(sym, 60.0)
            # Weight mentions by post virality: each mention contributes
            # 1 + log10(1 + upvotes). A 0-upvote post counts as 1; a 1k-upvote
            # post counts as ~4; a 10k-upvote post counts as ~5. Diminishing
            # returns prevent one viral post from drowning out broad coverage.
            effective_mentions = max(
                1,
                int(round(mentions * (1.0 + math.log10(1.0 + upvotes / max(1, mentions))))),
            )
            buzz = self._scorer.score(effective_mentions, "reddit", age_min)
            # Cap evidence at three rows so the prompt stays compact.
            ev = evidence_by_symbol.get(sym, [])[:3]
            out.append(
                DiscoveredTicker(
                    symbol=sym,
                    mention_count=mentions,
                    source_weight=BuzzScorer.SOURCE_WEIGHTS["reddit"],
                    recency_minutes=age_min,
                    buzz_score=buzz,
                    discovered_at=now,
                    evidence=ev,
                )
            )

        out.sort(key=lambda d: d.buzz_score, reverse=True)
        return out


def _parse_listing(payload: object, sub: str) -> list[_Post]:
    """Project Reddit's nested listing shape into trimmed ``_Post`` rows.

    Reddit listings are always ``{"data": {"children": [{"data": {...}}, ...]}}``
    — but JSON is JSON, so be defensive at every level. Bad rows are dropped.
    """
    out: list[_Post] = []
    if not isinstance(payload, dict):
        return out
    data = payload.get("data")
    if not isinstance(data, dict):
        return out
    children = data.get("children")
    if not isinstance(children, list):
        return out
    for child in children:
        if not isinstance(child, dict):
            continue
        post = child.get("data")
        if not isinstance(post, dict):
            continue
        try:
            out.append(
                _Post(
                    title=str(post.get("title", "")),
                    selftext=str(post.get("selftext", "")),
                    score=int(post.get("score", 0) or 0),
                    created_utc=float(post.get("created_utc", 0.0) or 0.0),
                    subreddit=str(post.get("subreddit", sub)),
                    permalink=str(post.get("permalink", "")),
                )
            )
        except (TypeError, ValueError):
            # Malformed row — keep going.
            continue
    return out
