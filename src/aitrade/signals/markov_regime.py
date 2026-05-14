"""Markov-chain regime classifier.

A 3-state, observable-state Markov chain over discretized (return, vol)
features. NOT a hidden-Markov model — we use observable buckets so the
classifier is interpretable, fast, and doesn't need scipy/hmmlearn.

States (semantic, applied per bar):

  0 = TRENDING_UP    20d return > +5%  AND  20d rvol annualized < 25%
  1 = MEAN_REVERTING (the residual / mid-volatility, modest-return regime)
  2 = STRESSED       20d return < −5%  OR   20d rvol annualized > 35%

The two thresholds (return ± 5% over 20 days, rvol 25 / 35 %) come from
US-equity sample percentiles. The middle bucket (MEAN_REVERTING) is the
modal regime by design — that's where bollinger has its edge per
EXP-001 / EXP-007.

What this gives us beyond a plain "rvol percentile" gate:

  - **Transition probabilities.** Given current state X, P(next = Y).
    Fitted as the empirical fraction of historical X→Y transitions.
    A bar that's currently STRESSED but whose transition matrix shows
    P(MEAN_REVERTING) = 0.7 is informationally different from one
    where P(STRESSED) = 0.7 (autoregressive stress).
  - **Steady-state mix.** The stationary distribution of the chain
    tells us the long-run regime mix on this symbol — useful for
    sizing and for the LLM reasoner's situational awareness.
  - **State sequence history.** The full path of states is logged so
    the LLM and the calibration harness can see what regime each
    historical trade was taken in.

This module is independent of the Strategy Protocol. It can be used as:
  - a conviction-component scorer (`signals/components/markov_regime.py`)
  - a stand-alone feature in the engine's per-candidate dump
  - a journal-event payload for offline analysis
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

from aitrade.data.models import Bar

# Tuning constants — kept here, not exposed as scorer params, because
# they have semantic meaning ("trending = +5% over 20d") and changing
# them changes what the state labels MEAN. Don't grid-search these.
_RETURN_WINDOW = 20
_VOL_WINDOW = 20
_TREND_RETURN = 0.05
_STRESS_RETURN = -0.05
_LOW_VOL_ANN = 0.25
_HIGH_VOL_ANN = 0.35

# Minimum bars before classification is reliable.
MIN_BARS_FOR_CLASSIFICATION = _RETURN_WINDOW + 1


class Regime(IntEnum):
    TRENDING_UP = 0
    MEAN_REVERTING = 1
    STRESSED = 2


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Frozen snapshot of the chain at a given point in time."""

    current_state: Regime
    state_history: tuple[Regime, ...]
    transition_matrix: tuple[tuple[float, float, float], ...]
    stationary_distribution: tuple[float, float, float]
    return_20d: float
    rvol_20d_annualized: float

    def prob_next_state(self, target: Regime) -> float:
        """P(next bar state = target | current state)."""
        i = int(self.current_state)
        j = int(target)
        return self.transition_matrix[i][j]


def _annualized_rvol(closes: Sequence[float]) -> float:
    """Daily log-return stdev × sqrt(252). Returns 0 on degenerate input."""
    if len(closes) < 2:
        return 0.0
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(rets)
    if n < 2:
        return 0.0
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def classify_state(return_20d: float, rvol_ann: float) -> Regime:
    """Apply the static decision rule. Pure function — no history."""
    if return_20d < _STRESS_RETURN or rvol_ann > _HIGH_VOL_ANN:
        return Regime.STRESSED
    if return_20d > _TREND_RETURN and rvol_ann < _LOW_VOL_ANN:
        return Regime.TRENDING_UP
    return Regime.MEAN_REVERTING


def state_sequence(bars: Sequence[Bar]) -> list[Regime]:
    """Produce one Regime label per bar (after the warmup window)."""
    closes = [b.close for b in bars]
    out: list[Regime] = []
    for i in range(len(closes)):
        if i < _RETURN_WINDOW:
            continue
        window = closes[i - _RETURN_WINDOW : i + 1]
        return_20d = window[-1] / window[0] - 1.0
        rvol_ann = _annualized_rvol(window)
        out.append(classify_state(return_20d, rvol_ann))
    return out


def fit_transition_matrix(
    states: Sequence[Regime],
) -> tuple[tuple[float, float, float], ...]:
    """Empirical transition matrix from a state sequence.

    Rows sum to 1.0. Uses Laplace smoothing (add-1) to avoid zero-prob
    rows on small samples — a regime that's never been seen still gets
    a non-zero estimate, which the LLM can read with the right caveat.
    """
    counts = [[1, 1, 1] for _ in range(3)]
    for a, b in zip(states[:-1], states[1:], strict=False):
        counts[int(a)][int(b)] += 1
    rows: list[tuple[float, float, float]] = []
    for row in counts:
        total = float(sum(row))
        rows.append((row[0] / total, row[1] / total, row[2] / total))
    return (rows[0], rows[1], rows[2])


def stationary_distribution(
    transition_matrix: tuple[tuple[float, float, float], ...],
    iterations: int = 200,
) -> tuple[float, float, float]:
    """Power-iterate to the chain's stationary distribution.

    Doesn't need numpy — three states, 200 iterations of (1×3)·(3×3)
    is trivially fast and exact enough for our purposes.
    """
    dist = [1 / 3, 1 / 3, 1 / 3]
    for _ in range(iterations):
        new_dist = [
            sum(dist[i] * transition_matrix[i][j] for i in range(3))
            for j in range(3)
        ]
        # Renormalize each step against floating drift.
        s = sum(new_dist)
        new_dist = [x / s for x in new_dist] if s > 0 else dist
        dist = new_dist
    return (dist[0], dist[1], dist[2])


@dataclass
class MarkovRegimeChain:
    """Stateful chain — incrementally absorbs bars and exposes snapshots.

    Call ``observe(bar)`` for each bar in chronological order. Once
    enough history is present, ``snapshot()`` returns the current state,
    fitted transition matrix, and stationary distribution.

    Memory is bounded by ``max_history`` (default 1000 bars ≈ 4 years
    of daily data) to keep the transition fit recent.
    """

    max_history: int = 1000
    _closes: list[float] = field(init=False, default_factory=list)
    _states: list[Regime] = field(init=False, default_factory=list)

    def observe(self, bar: Bar) -> None:
        self._closes.append(bar.close)
        if len(self._closes) > self.max_history:
            self._closes = self._closes[-self.max_history :]
        if len(self._closes) > _RETURN_WINDOW:
            window = self._closes[-(_RETURN_WINDOW + 1) :]
            return_20d = window[-1] / window[0] - 1.0
            rvol_ann = _annualized_rvol(window)
            self._states.append(classify_state(return_20d, rvol_ann))
            if len(self._states) > self.max_history:
                self._states = self._states[-self.max_history :]

    def snapshot(self) -> RegimeSnapshot | None:
        if len(self._states) < 2:
            return None
        matrix = fit_transition_matrix(self._states)
        steady = stationary_distribution(matrix)
        window = self._closes[-(_RETURN_WINDOW + 1) :]
        return_20d = window[-1] / window[0] - 1.0 if len(window) > 1 else 0.0
        rvol_ann = _annualized_rvol(window) if len(window) > 1 else 0.0
        return RegimeSnapshot(
            current_state=self._states[-1],
            state_history=tuple(self._states),
            transition_matrix=matrix,
            stationary_distribution=steady,
            return_20d=return_20d,
            rvol_20d_annualized=rvol_ann,
        )
