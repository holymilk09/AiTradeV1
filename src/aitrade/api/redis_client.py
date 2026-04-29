"""Async Redis client singleton."""

from __future__ import annotations

from functools import lru_cache

from redis.asyncio import Redis

from aitrade.api.settings import get_settings


@lru_cache(maxsize=1)
def redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)
