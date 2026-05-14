"""Scorer Protocol + ScoringContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aitrade.data.models import Bar
from aitrade.signals.conviction import ConvictionComponent
from aitrade.strategy.signal import Signal


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """Inputs every scorer sees.

    Designed minimally — anything else a scorer needs (pattern detectors,
    trained model, prompt) is held on the scorer instance, not in context.
    """

    bars: list[Bar]  # chronological, most recent last
    signal: Signal | None  # current strategy signal, if any


@runtime_checkable
class Scorer(Protocol):
    """A component scorer. Returns a ConvictionComponent."""

    name: str
    default_weight: float

    def score(self, ctx: ScoringContext) -> ConvictionComponent: ...
