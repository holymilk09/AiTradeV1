"""Conviction types + composite calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ConvictionComponent:
    """One scorer's output. score and weight are in [0, 1]."""

    score: float
    weight: float
    evidence: str
    features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConvictionSnapshot:
    """All component outputs + composite score for one (ticker, as_of)."""

    ticker: str
    as_of: datetime
    composite: float
    components: dict[str, ConvictionComponent]


def composite_from_components(
    components: dict[str, ConvictionComponent],
) -> float:
    """Weighted sum normalized by total weight. Returns 0 if all weights are 0."""
    total_weight = sum(c.weight for c in components.values())
    if total_weight <= 0:
        return 0.0
    return sum(c.score * c.weight for c in components.values()) / total_weight
