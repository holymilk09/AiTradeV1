"""Tests for the BuzzScorer formula and source classification."""

from __future__ import annotations

import math

from aitrade.discovery.scorer import BuzzScorer, DiscoveredTicker


def test_score_formula_basic() -> None:
    """Spot-check the closed-form: log(1+m) * weight * exp(-age/60)."""
    scorer = BuzzScorer()
    # 10 mentions on Bloomberg (weight 1.0) at age 0 → log(11) * 1.0 * 1.0
    score = scorer.score(mention_count=10, source="bloomberg", age_minutes=0.0)
    assert math.isclose(score, math.log(11), rel_tol=1e-6)


def test_score_zero_mentions_yields_zero() -> None:
    """log(1 + 0) == 0 → score must be 0 regardless of source / age."""
    scorer = BuzzScorer()
    assert scorer.score(0, "bloomberg", 0.0) == 0.0
    assert scorer.score(0, "reddit", 100.0) == 0.0


def test_score_unknown_source_uses_unknown_weight() -> None:
    """Sources not in SOURCE_WEIGHTS must fall back to the 'unknown' weight."""
    scorer = BuzzScorer()
    score = scorer.score(5, "some-random-source", 0.0)
    expected = math.log(6) * 0.3 * 1.0
    assert math.isclose(score, expected, rel_tol=1e-6)


def test_classify_source_known_urls() -> None:
    scorer = BuzzScorer()
    assert scorer.classify_source("https://www.bloomberg.com/news/article") == "bloomberg"
    assert scorer.classify_source("https://www.cnbc.com/quotes/AAPL") == "cnbc"
    assert scorer.classify_source("https://reddit.com/r/wallstreetbets") == "reddit"
    assert scorer.classify_source("https://twitter.com/elonmusk") == "twitter"


def test_classify_source_case_insensitive() -> None:
    scorer = BuzzScorer()
    assert scorer.classify_source("HTTPS://WWW.BLOOMBERG.COM/x") == "bloomberg"
    assert scorer.classify_source("https://Reuters.com/Markets") == "reuters"


def test_classify_source_unknown_returns_unknown() -> None:
    scorer = BuzzScorer()
    assert scorer.classify_source("https://random-blog.example.org/post") == "unknown"
    assert scorer.classify_source("") == "unknown"


def test_recency_decay_older_scores_lower() -> None:
    """Older content must always score below newer content, all else equal."""
    scorer = BuzzScorer()
    fresh = scorer.score(mention_count=5, source="reuters", age_minutes=0.0)
    one_hour = scorer.score(mention_count=5, source="reuters", age_minutes=60.0)
    one_day = scorer.score(mention_count=5, source="reuters", age_minutes=24 * 60.0)
    assert fresh > one_hour > one_day
    # 60-minute half-time-constant: at 60 min, score should be ~1/e of fresh.
    assert math.isclose(one_hour / fresh, math.exp(-1.0), rel_tol=1e-6)


def test_higher_credibility_source_outscores_lower() -> None:
    """Bloomberg (1.0) must beat Reddit (0.4) at equal mentions and age."""
    scorer = BuzzScorer()
    bloom = scorer.score(10, "bloomberg", 30.0)
    reddit = scorer.score(10, "reddit", 30.0)
    assert bloom > reddit


def test_discovered_ticker_is_frozen() -> None:
    """The dataclass is frozen — attribute assignment must raise."""
    from dataclasses import FrozenInstanceError
    from datetime import UTC, datetime

    dt = DiscoveredTicker(
        symbol="AAPL",
        mention_count=5,
        source_weight=1.0,
        recency_minutes=10.0,
        buzz_score=1.5,
        discovered_at=datetime.now(UTC),
        evidence=["context snippet"],
    )
    try:
        dt.symbol = "TSLA"  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("DiscoveredTicker must be frozen")
