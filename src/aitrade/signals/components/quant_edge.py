"""quant_edge: derived from the strategy's own signal strength."""

from __future__ import annotations

from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent


class QuantEdgeScorer:
    name: str = "quant_edge"
    default_weight: float = 0.4

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        if ctx.signal is None:
            return ConvictionComponent(
                score=0.0,
                weight=self.default_weight,
                evidence="no signal",
                features={"strength": 0.0},
            )
        s = max(0.0, min(1.0, ctx.signal.strength))
        return ConvictionComponent(
            score=s,
            weight=self.default_weight,
            evidence=ctx.signal.reason or f"strategy strength={s:.2f}",
            features={"strength": s},
        )
