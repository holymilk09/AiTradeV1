"""Buzz scoring — combine mention volume, source credibility, and recency.

The score is a simple closed-form expression so it's easy to reason about and
test:

.. code:: text

    buzz = log(1 + mentions) * source_weight * exp(-age_minutes / 60)

* ``log(1 + mentions)`` — saturates so ten thousand tweets don't drown out one
  Bloomberg article.
* ``source_weight`` — credibility prior; tier-1 financial press at 1.0, social
  at 0.4, unknown at 0.3.
* ``exp(-age / 60)`` — half-life of ~42 minutes; a one-hour-old story is worth
  ~37% of a fresh one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class DiscoveredTicker:
    """A symbol surfaced by the discovery agent, scored by buzz.

    Attributes
    ----------
    symbol:
        Validated ticker (uppercase, in the tradable universe).
    mention_count:
        How many times the symbol appeared in source material.
    source_weight:
        Credibility multiplier for the originating source (see
        :attr:`BuzzScorer.SOURCE_WEIGHTS`).
    recency_minutes:
        Age of the source material at the time of scoring, in minutes.
    buzz_score:
        Composite score; higher = more buzzy. See module docstring for formula.
    evidence:
        Short context snippets that triggered the mention. Useful for
        downstream prompts and for human spot-checks.
    discovered_at:
        Wall-clock time the agent surfaced this ticker.
    """

    symbol: str
    mention_count: int
    source_weight: float
    recency_minutes: float
    buzz_score: float
    discovered_at: datetime
    evidence: list[str] = field(default_factory=list)


class BuzzScorer:
    """Compose mention count, source credibility, and recency into one score."""

    SOURCE_WEIGHTS: ClassVar[dict[str, float]] = {
        "bloomberg": 1.0,
        "wsj": 1.0,
        "ft": 1.0,
        "reuters": 0.8,
        "cnbc": 0.8,
        "yahoo": 0.6,
        "marketwatch": 0.6,
        "seekingalpha": 0.6,
        "reddit": 0.4,
        "twitter": 0.4,
        "stocktwits": 0.4,
        "x.com": 0.4,
        "unknown": 0.3,
    }

    def score(self, mention_count: int, source: str, age_minutes: float) -> float:
        """Return the composite buzz score.

        Parameters
        ----------
        mention_count:
            Number of mentions; passed through ``log(1 + n)`` so the curve
            saturates and isn't dominated by viral tweets.
        source:
            A key from :attr:`SOURCE_WEIGHTS`. Unknown keys fall back to the
            ``"unknown"`` weight (0.3) — be conservative with un-classified
            sources rather than discarding them.
        age_minutes:
            Age of the source content. Decays with a 60-minute time constant.
        """
        weight = self.SOURCE_WEIGHTS.get(source, self.SOURCE_WEIGHTS["unknown"])
        return math.log(1 + mention_count) * weight * math.exp(-age_minutes / 60.0)

    def classify_source(self, url: str) -> str:
        """Map a URL to a known source key by case-insensitive substring match.

        Returns ``"unknown"`` if no known source key appears in the URL. The
        match is order-independent — first hit wins; collisions between known
        keys are not expected because the keys are distinct domain fragments.
        """
        url_lower = url.lower()
        for key in self.SOURCE_WEIGHTS:
            if key == "unknown":
                continue
            if key in url_lower:
                return key
        return "unknown"
