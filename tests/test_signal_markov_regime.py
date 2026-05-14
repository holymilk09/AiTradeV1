"""Markov regime chain + scorer.

Synthetic series used here are *test fixtures only* — designed to drive
deterministic state transitions for assertions. Never used in research.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from aitrade.data.models import Bar
from aitrade.signals.components import MarkovRegimeScorer, ScoringContext
from aitrade.signals.markov_regime import (
    MarkovRegimeChain,
    Regime,
    classify_state,
    fit_transition_matrix,
    state_sequence,
    stationary_distribution,
)


def _bar(i: int, close: float) -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
        open=close, high=close + 0.1, low=close - 0.1, close=close, volume=1_000_000,
    )


def test_classify_state_thresholds() -> None:
    # Trending: return > +5% AND vol < 25%
    assert classify_state(0.10, 0.18) is Regime.TRENDING_UP
    # Stressed: return < -5%
    assert classify_state(-0.07, 0.20) is Regime.STRESSED
    # Stressed: vol > 35%
    assert classify_state(0.02, 0.45) is Regime.STRESSED
    # Mean reverting: middle bucket
    assert classify_state(0.02, 0.20) is Regime.MEAN_REVERTING
    # Edge case: high return but high vol → mean_reverting (fails trending check)
    assert classify_state(0.10, 0.30) is Regime.MEAN_REVERTING


def test_transition_matrix_rows_sum_to_one() -> None:
    states = [Regime.MEAN_REVERTING] * 20 + [Regime.STRESSED] * 5 + [Regime.TRENDING_UP] * 10
    m = fit_transition_matrix(states)
    for row in m:
        assert abs(sum(row) - 1.0) < 1e-9


def test_stationary_distribution_sums_to_one() -> None:
    m = (
        (0.8, 0.15, 0.05),
        (0.10, 0.80, 0.10),
        (0.05, 0.15, 0.80),
    )
    s = stationary_distribution(m, iterations=300)
    assert abs(sum(s) - 1.0) < 1e-6
    # Identity-leaning chain → roughly even split
    for v in s:
        assert 0.2 < v < 0.5


def test_state_sequence_uptrend_then_crash() -> None:
    """Construct a long uptrend (low daily move) → stressed crash. Verify
    the labeled state path matches the construction."""
    bars: list[Bar] = []
    # 60 calm uptrend bars: ~0.5%/day with no noise → 20-bar return ~10%, vol low.
    price = 100.0
    for i in range(60):
        price *= math.exp(0.005)
        bars.append(_bar(i, price))
    # 30 crash bars: −2%/day → 20-bar return < −5%.
    for i in range(60, 90):
        price *= math.exp(-0.02)
        bars.append(_bar(i, price))
    seq = state_sequence(bars)
    # First labels (post-warmup) should be trending; last labels stressed.
    assert seq[0] is Regime.TRENDING_UP
    assert seq[-1] is Regime.STRESSED


def test_chain_observe_and_snapshot() -> None:
    chain = MarkovRegimeChain()
    # Feed 30 calm-trending bars.
    price = 100.0
    for i in range(30):
        price *= math.exp(0.004)
        chain.observe(_bar(i, price))
    snap = chain.snapshot()
    assert snap is not None
    assert snap.current_state is Regime.TRENDING_UP
    # Transition matrix exists with valid rows.
    assert len(snap.transition_matrix) == 3
    for row in snap.transition_matrix:
        assert abs(sum(row) - 1.0) < 1e-9


def test_scorer_insufficient_history_returns_neutral() -> None:
    scorer = MarkovRegimeScorer()
    bars = [_bar(i, 100.0 + i * 0.1) for i in range(5)]
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.score == 0.5
    assert "insufficient" in c.evidence


def test_scorer_trending_state_returns_mid_score() -> None:
    scorer = MarkovRegimeScorer()
    bars: list[Bar] = []
    price = 100.0
    for i in range(60):
        price *= math.exp(0.005)  # trending up, low vol
        bars.append(_bar(i, price))
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.features["state"] == float(int(Regime.TRENDING_UP))
    # TRENDING_UP base score is 0.5; may flex slightly with transitions.
    assert 0.3 <= c.score <= 0.8


def test_scorer_stressed_state_returns_low_score() -> None:
    scorer = MarkovRegimeScorer()
    bars: list[Bar] = []
    price = 100.0
    # 30 trending bars then 30 crashing bars → ends in stressed.
    for i in range(30):
        price *= math.exp(0.005)
        bars.append(_bar(i, price))
    for i in range(30, 60):
        price *= math.exp(-0.02)
        bars.append(_bar(i, price))
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    assert c.features["state"] == float(int(Regime.STRESSED))
    # STRESSED base 0.0; with high P(→mean_reverting) it could lift but
    # in this synthetic crash the chain hasn't seen a revert yet, so
    # the boost is small.
    assert c.score < 0.5


def test_scorer_features_carry_transition_probabilities() -> None:
    scorer = MarkovRegimeScorer()
    bars = [_bar(i, 100.0 + i * 0.2) for i in range(80)]
    c = scorer.score(ScoringContext(bars=bars, signal=None))
    for key in (
        "state",
        "return_20d",
        "rvol_20d",
        "p_next_mean_reverting",
        "p_next_stressed",
        "stationary_mean_reverting",
        "stationary_stressed",
    ):
        assert key in c.features
    # Probabilities are in [0, 1].
    for k in ("p_next_mean_reverting", "p_next_stressed",
              "stationary_mean_reverting", "stationary_stressed"):
        assert 0.0 <= c.features[k] <= 1.0
