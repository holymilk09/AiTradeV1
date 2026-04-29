"""Async Redis client singleton."""

from __future__ import annotations

from redis.asyncio import Redis

from aitrade.api.settings import get_settings

_client: Redis | None = None


def redis() -> Redis:
    global _client
    if _client is None:
        _client = Redis.from_url(get_settings().redis_url, decode_responses=True)
    return _client
