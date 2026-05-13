"""llm_reasoning: placeholder. Real implementation lands in M4.

For M3, defaults to weight=0 so it does not influence composite scoring
until proven valuable. Callers that wire a reasoner in can pass a custom
score via the ``override_score`` constructor arg for offline replay.
"""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent


@dataclass
class LlmReasoningScorer:
    name: str = "llm_reasoning"
    default_weight: float = 0.0  # Off until M4 validates lift.
    override_score: float | None = None
    evidence_text: str = "M4 placeholder"

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        s = self.override_score if self.override_score is not None else 0.5
        return ConvictionComponent(
            score=max(0.0, min(1.0, s)),
            weight=self.default_weight,
            evidence=self.evidence_text,
        )
