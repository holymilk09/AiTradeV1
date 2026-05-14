"""markov_regime: regime-state Markov-chain conviction component.

Wraps :class:`MarkovRegimeChain` for the existing M3 conviction-component
contract. The score encodes "favorable regime for the strategy that
emitted the signal" — for the current registry, that means **favorable
for mean-reversion**.

Scoring:
  - Current state == MEAN_REVERTING                  → score 1.0
  - Current state == TRENDING_UP                     → score 0.5
  - Current state == STRESSED                        → score 0.0
  - But also: a STRESSED bar with a HIGH transition probability into
    MEAN_REVERTING gets a partial boost — the chain is "about to
    revert", and a forward-looking trader can lean into that.

Features exposed (so the LLM reasoner can read them directly):
  - state                          : "trending_up" | "mean_reverting" | "stressed"
  - return_20d                     : the underlying 20-bar return
  - rvol_20d                       : annualized 20-bar realized vol
  - p_next_mean_reverting          : P(next bar = MEAN_REVERTING | current)
  - p_next_stressed                : P(next bar = STRESSED | current)
  - stationary_mean_reverting      : long-run fraction in MEAN_REVERTING
  - stationary_stressed            : long-run fraction in STRESSED
"""

from __future__ import annotations

from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent
from aitrade.signals.markov_regime import (
    MIN_BARS_FOR_CLASSIFICATION,
    MarkovRegimeChain,
    Regime,
)

_STATE_NAME = {
    Regime.TRENDING_UP: "trending_up",
    Regime.MEAN_REVERTING: "mean_reverting",
    Regime.STRESSED: "stressed",
}
_BASE_SCORE = {
    Regime.MEAN_REVERTING: 1.0,
    Regime.TRENDING_UP: 0.5,
    Regime.STRESSED: 0.0,
}


class MarkovRegimeScorer:
    """Markov-chain regime conviction component.

    Each call rebuilds the chain from the bars in the ScoringContext.
    Stateless on the scorer side — the chain state lives on the bars
    you pass in. That's the right design for a backtest harness where
    each fold creates a fresh strategy + scorer.
    """

    name: str = "markov_regime"
    default_weight: float = 0.25

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        bars = ctx.bars
        if len(bars) < MIN_BARS_FOR_CLASSIFICATION + 1:
            return ConvictionComponent(
                score=0.5,
                weight=self.default_weight,
                evidence=f"insufficient history ({len(bars)} bars)",
                features={"state": -1.0},
            )

        chain = MarkovRegimeChain()
        for b in bars:
            chain.observe(b)
        snap = chain.snapshot()
        if snap is None:
            return ConvictionComponent(
                score=0.5,
                weight=self.default_weight,
                evidence="chain produced no snapshot",
                features={"state": -1.0},
            )

        base = _BASE_SCORE[snap.current_state]
        p_revert = snap.prob_next_state(Regime.MEAN_REVERTING)
        p_stress = snap.prob_next_state(Regime.STRESSED)

        # If we're STRESSED but heading into MEAN_REVERTING, boost.
        if snap.current_state is Regime.STRESSED and p_revert > 0.5:
            score = base + (p_revert - 0.5) * 0.6  # up to +0.3 boost
        # If we're MEAN_REVERTING but P(STRESSED) is high, fade.
        elif snap.current_state is Regime.MEAN_REVERTING and p_stress > 0.4:
            score = base - (p_stress - 0.4) * 0.8  # up to −0.5 fade
        else:
            score = base
        score = max(0.0, min(1.0, score))

        state_name = _STATE_NAME[snap.current_state]
        # Features must be float per ConvictionComponent contract. State
        # is encoded as the IntEnum value (0=TRENDING_UP, 1=MEAN_REVERTING,
        # 2=STRESSED); the human-readable label lives in `evidence`.
        return ConvictionComponent(
            score=score,
            weight=self.default_weight,
            evidence=(
                f"regime={state_name} ret20={snap.return_20d:+.1%} "
                f"rvol20={snap.rvol_20d_annualized:.1%} "
                f"P(→mean_rev)={p_revert:.2f} P(→stress)={p_stress:.2f}"
            ),
            features={
                "state": float(int(snap.current_state)),
                "return_20d": round(snap.return_20d, 4),
                "rvol_20d": round(snap.rvol_20d_annualized, 4),
                "p_next_mean_reverting": round(p_revert, 3),
                "p_next_stressed": round(p_stress, 3),
                "stationary_mean_reverting": round(snap.stationary_distribution[1], 3),
                "stationary_stressed": round(snap.stationary_distribution[2], 3),
            },
        )
