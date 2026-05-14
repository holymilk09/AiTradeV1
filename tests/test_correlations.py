"""Phase 8 — pairwise correlation math + stacking penalty."""

from __future__ import annotations

import math

import pytest

from aitrade.market.correlations import (
    CorrelationMatrix,
    closes_to_log_returns,
    compute_correlation_matrix,
    correlation_penalty_for,
    pearson,
)

# --------------------------------------------------------------------------- #
# closes_to_log_returns                                                        #
# --------------------------------------------------------------------------- #


def test_log_returns_basic_sequence() -> None:
    closes = [100.0, 105.0, 102.9, 105.0]  # +5%, -2%, +2%
    rs = closes_to_log_returns(closes)
    assert len(rs) == 3
    assert rs[0] == pytest.approx(math.log(1.05))
    assert rs[2] == pytest.approx(math.log(105 / 102.9))


def test_log_returns_skips_non_positive() -> None:
    """Zero / negative prices in the middle of a series are skipped, not
    returned as NaN, so callers don't pollute their downstream stats."""
    rs = closes_to_log_returns([100.0, 0.0, 110.0, -5.0, 120.0])
    # Only valid transitions: 100→? (skipped), ?→110 from 0 (skipped),
    # 110→-5 (skipped), -5→120 (skipped). All have a non-positive endpoint.
    assert rs == []


# --------------------------------------------------------------------------- #
# pearson                                                                      #
# --------------------------------------------------------------------------- #


def test_pearson_perfectly_correlated() -> None:
    # Same series ⇒ ρ = 1.0
    series = [0.01 * i for i in range(60)]
    assert pearson(series, series) == pytest.approx(1.0)


def test_pearson_perfectly_anticorrelated() -> None:
    series = [0.01 * i for i in range(60)]
    inverted = [-x for x in series]
    assert pearson(series, inverted) == pytest.approx(-1.0)


def test_pearson_below_min_overlap_returns_none() -> None:
    short = [0.01, 0.02, 0.03, -0.01, 0.0]  # < 30 points
    assert pearson(short, short) is None


def test_pearson_constant_input_returns_none() -> None:
    constant = [0.0] * 60
    other = [0.01 * i for i in range(60)]
    # Zero-variance input → undefined ρ.
    assert pearson(constant, other) is None


def test_pearson_uncorrelated_random_walk() -> None:
    """Two independent linear signals with different phases land near zero,
    not at ±1. Catches sign-flip / scale bugs in the formula."""
    a = [math.sin(0.1 * i) for i in range(120)]
    b = [math.cos(0.13 * i) for i in range(120)]
    rho = pearson(a, b)
    assert rho is not None
    assert abs(rho) < 0.5


# --------------------------------------------------------------------------- #
# compute_correlation_matrix                                                   #
# --------------------------------------------------------------------------- #


def test_matrix_omits_undercomputed_pairs() -> None:
    """Pairs whose overlap is below the floor are listed in
    `insufficient_pairs` rather than being given a fake 0.0."""
    long_run = [0.01 * i for i in range(60)]
    short_run = [0.01, 0.02]  # too short
    mat = compute_correlation_matrix({"AAA": long_run, "BBB": short_run})
    assert ("AAA", "BBB") in mat.insufficient_pairs
    assert ("AAA", "BBB") not in mat.pairs
    assert mat.symbols == ["AAA", "BBB"]


def test_matrix_self_lookup_returns_one() -> None:
    long_run = [0.01 * i for i in range(60)]
    mat = compute_correlation_matrix({"AAA": long_run, "BBB": long_run})
    assert mat.get("AAA", "AAA") == 1.0
    # Symmetric lookup
    assert mat.get("AAA", "BBB") == mat.get("BBB", "AAA")


def test_matrix_to_journal_payload_round_trips_fields() -> None:
    long_run = [0.01 * i for i in range(60)]
    mat = compute_correlation_matrix({"AAA": long_run, "BBB": long_run})
    payload = mat.to_journal_payload()
    assert payload["symbols"] == ["AAA", "BBB"]
    assert isinstance(payload["pairs"], list)
    assert payload["pairs"][0]["a"] == "AAA"
    assert payload["pairs"][0]["b"] == "BBB"
    assert "rho" in payload["pairs"][0]


# --------------------------------------------------------------------------- #
# correlation_penalty_for                                                      #
# --------------------------------------------------------------------------- #


def _matrix_with_pairs(
    pairs: dict[tuple[str, str], float],
) -> CorrelationMatrix:
    """Build a CorrelationMatrix directly from a {(a,b): rho} dict for test
    ergonomics — bypasses the math so we can target the penalty rule."""
    symbols = sorted({s for pair in pairs for s in pair})
    normalized = {
        (a, b) if a < b else (b, a): rho for (a, b), rho in pairs.items()
    }
    return CorrelationMatrix(
        symbols=symbols, pairs=normalized, insufficient_pairs=[]
    )


def test_penalty_no_held_positions_is_zero() -> None:
    mat = _matrix_with_pairs({("AAA", "BBB"): 0.95})
    assert correlation_penalty_for(mat, "AAA", []) == 0.0


def test_penalty_high_correlation_is_minus_half() -> None:
    mat = _matrix_with_pairs({("CAND", "HELD"): 0.85})
    assert correlation_penalty_for(mat, "CAND", ["HELD"]) == -0.5


def test_penalty_medium_correlation_is_minus_quarter() -> None:
    mat = _matrix_with_pairs({("CAND", "HELD"): 0.55})
    assert correlation_penalty_for(mat, "CAND", ["HELD"]) == -0.25


def test_penalty_low_correlation_is_zero() -> None:
    mat = _matrix_with_pairs({("CAND", "HELD"): 0.20})
    assert correlation_penalty_for(mat, "CAND", ["HELD"]) == 0.0


def test_penalty_uses_max_abs_across_held() -> None:
    """If the candidate is mildly correlated with one held position but
    *very* anticorrelated with another, the penalty fires off the bigger
    absolute value — both directions represent stacked exposure."""
    mat = _matrix_with_pairs(
        {("CAND", "H1"): 0.30, ("CAND", "H2"): -0.85}
    )
    assert correlation_penalty_for(mat, "CAND", ["H1", "H2"]) == -0.5


def test_penalty_self_held_does_not_self_stack() -> None:
    """A held-position symbol asking for its own penalty returns 0 — the
    position-fit bonus already handles the stacking-into-self case."""
    mat = _matrix_with_pairs({("AAA", "BBB"): 0.95})
    assert correlation_penalty_for(mat, "AAA", ["AAA"]) == 0.0


def test_penalty_skips_undercomputed_pairs() -> None:
    """If we don't have a ρ for the candidate-vs-held pair (insufficient
    overlap), the candidate gets 0 penalty rather than a falsely-zero one."""
    mat = CorrelationMatrix(
        symbols=["CAND", "HELD"],
        pairs={},  # no pairs computed
        insufficient_pairs=[("CAND", "HELD")],
    )
    assert correlation_penalty_for(mat, "CAND", ["HELD"]) == 0.0
