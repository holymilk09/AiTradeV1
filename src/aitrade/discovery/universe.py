"""Universe = watchlist (always) + movers (always) + reddit (opt-in) + web (opt-in).

The engine calls :func:`build_universe` each cycle to get the merged list of
``DiscoveredTicker`` candidates. Sources compose by symbol — if AAPL shows
up in both watchlist (buzz=1.0) and movers (buzz=0.7), the higher wins; the
evidence snippets accumulate so the reasoner sees both signals.

Composition order (deliberate):

  1. Watchlist — fixed, free, deterministic. Always there.
  2. Movers    — Alpaca screener; fast, free, deterministic.
  3. Reddit    — public ``.json`` endpoint; opt-in. Adds retail-trader buzz
                 (r/wallstreetbets, r/stocks, r/options).
  4. Web       — ``DiscoveryAgent`` (Claude + web_search). Opt-in only —
                 noisy, slow, rate-limit-prone, can drop on flaky networks.

If reddit or web is enabled and fails (typical: rate limit, VPN-killed stream),
the engine still runs from watchlist + movers. The bot never fails just
because Reddit or Twitter is having a bad day.
"""

from __future__ import annotations

from aitrade.discovery.agent import DiscoveryAgent
from aitrade.discovery.movers import MoversFinder
from aitrade.discovery.reddit import RedditDiscoveryClient
from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.discovery.watchlist import watchlist_as_discovered


def merge_sources(*sources: list[DiscoveredTicker]) -> list[DiscoveredTicker]:
    """Union DiscoveredTicker lists by symbol; higher buzz_score wins.

    Evidence snippets from all sources are concatenated on the winner so the
    reasoner sees the full picture (e.g. *"watchlist + most-active rank 3"*).
    """
    by_symbol: dict[str, DiscoveredTicker] = {}
    for source in sources:
        for t in source:
            existing = by_symbol.get(t.symbol)
            if existing is None:
                # Copy so we can extend evidence without mutating the input.
                by_symbol[t.symbol] = DiscoveredTicker(
                    symbol=t.symbol,
                    mention_count=t.mention_count,
                    source_weight=t.source_weight,
                    recency_minutes=t.recency_minutes,
                    buzz_score=t.buzz_score,
                    discovered_at=t.discovered_at,
                    evidence=list(t.evidence),
                )
                continue
            # Combine: higher buzz wins; mention_count adds; evidence concats.
            winner_buzz = max(existing.buzz_score, t.buzz_score)
            winner_weight = max(existing.source_weight, t.source_weight)
            winner_recency = min(existing.recency_minutes, t.recency_minutes)
            combined_evidence = list(existing.evidence)
            for snippet in t.evidence:
                if snippet not in combined_evidence:
                    combined_evidence.append(snippet)
            by_symbol[t.symbol] = DiscoveredTicker(
                symbol=t.symbol,
                mention_count=existing.mention_count + t.mention_count,
                source_weight=winner_weight,
                recency_minutes=winner_recency,
                buzz_score=winner_buzz,
                discovered_at=existing.discovered_at,
                evidence=combined_evidence,
            )
    out = list(by_symbol.values())
    out.sort(key=lambda d: d.buzz_score, reverse=True)
    return out


def build_universe(
    *,
    movers: MoversFinder | None = None,
    reddit: RedditDiscoveryClient | None = None,
    web_agent: DiscoveryAgent | None = None,
    web_top_n: int = 20,
    use_watchlist: bool = True,
    use_movers: bool = True,
    use_reddit: bool = False,
    use_web: bool = False,
) -> list[DiscoveredTicker]:
    """Compose the cycle's universe from the configured sources.

    Defaults: watchlist + movers ON, reddit + web OFF. The social/web layers
    are opt-in because they're network-dependent and can rate-limit; turn
    them on once you have a stable path and the right credentials.
    """
    sources: list[list[DiscoveredTicker]] = []

    if use_watchlist:
        sources.append(watchlist_as_discovered())

    if use_movers and movers is not None:
        sources.append(movers.find())

    if use_reddit and reddit is not None:
        sources.append(reddit.discover())

    if use_web and web_agent is not None:
        sources.append(web_agent.discover(top_n=web_top_n))

    return merge_sources(*sources)
