"""Base types for the pattern-detection layer.

Pattern detectors are pure functions over a window of recent bars. Each one
emits at most one :class:`PatternSignal` per call — never raising, never
mutating state — so the orchestrator can fan out across detectors cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from aitrade.data.models import Bar
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class PatternSignal:
    """A single pattern hit, with the numbers that justified it.

    ``score`` is always clamped to ``[0, 1]``; ``evidence`` carries the raw
    figures so the reasoner / journal can audit why the detector fired.
    """

    name: str
    symbol: str
    score: float
    direction: Direction
    timestamp: datetime
    evidence: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class PatternDetector(Protocol):
    """Runtime-checkable contract every detector must satisfy."""

    @property
    def name(self) -> str:
        """Stable identifier for the detector — used in logs and signals."""
        ...

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a :class:`PatternSignal` if the pattern fires, else None.

        Implementations must be pure: same bars in → same signal out, with no
        I/O and no mutation of ``bars``.
        """
        ...


def clamp01(x: float) -> float:
    """Clamp ``x`` to the inclusive ``[0.0, 1.0]`` range."""
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
