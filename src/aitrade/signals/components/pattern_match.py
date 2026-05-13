"""pattern_match: best confidence among registered detectors."""

from __future__ import annotations

from aitrade.patterns import PatternDetector, get_all_detectors
from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent


class PatternMatchScorer:
    name: str = "pattern_match"
    default_weight: float = 0.2

    def __init__(self, detectors: list[PatternDetector] | None = None) -> None:
        self._detectors = detectors if detectors is not None else get_all_detectors()

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        if not ctx.bars:
            return ConvictionComponent(
                score=0.0,
                weight=self.default_weight,
                evidence="no bars",
            )
        best_score = 0.0
        best_name = ""
        per_detector: dict[str, float] = {}
        for d in self._detectors:
            sig = d.detect(list(ctx.bars))
            if sig is None:
                per_detector[d.name] = 0.0
                continue
            per_detector[d.name] = sig.score
            if sig.score > best_score:
                best_score = sig.score
                best_name = d.name
        evidence = (
            f"best={best_name}@{best_score:.2f}" if best_name else "no pattern fired"
        )
        return ConvictionComponent(
            score=best_score,
            weight=self.default_weight,
            evidence=evidence,
            features=per_detector,
        )
