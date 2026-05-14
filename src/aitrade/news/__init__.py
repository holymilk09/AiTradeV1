"""News module — Alpaca News API wrapper with TTL cache + LLM-friendly projection."""

from __future__ import annotations

from aitrade.news.client import AlpacaNewsClient, NewsItem

__all__ = ["AlpacaNewsClient", "NewsItem"]
