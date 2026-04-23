"""Signal object + position sizing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    direction: Direction
    strength: float = 1.0  # 0..1 conviction
    reason: str = ""


def size_by_notional(*, notional_usd: float, price: float) -> float:
    """Round-down share count for a target notional. Minimum 1 share."""
    if price <= 0:
        raise ValueError("price must be positive")
    qty = notional_usd // price
    return max(qty, 1.0)
