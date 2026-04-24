"""Discovery module — find which tickers the market is buzzing about.

Uses Claude with web search/fetch tools to surface candidate symbols, validates
them against the Alpaca tradable universe, and scores them by buzz (mention
volume × source weight × recency decay).
"""

from __future__ import annotations

from aitrade.discovery.agent import DiscoveryAgent
from aitrade.discovery.extractor import TickerExtractor
from aitrade.discovery.scorer import BuzzScorer, DiscoveredTicker

__all__ = [
    "BuzzScorer",
    "DiscoveredTicker",
    "DiscoveryAgent",
    "TickerExtractor",
]
