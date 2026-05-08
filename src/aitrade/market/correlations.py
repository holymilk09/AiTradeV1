"""Phase 8 — pairwise Pearson correlation between symbols.

Used by the engine to penalize stacking correlated names on the
candidate board (long NVDA + AMD signal fires = 2x tech beta, not a
fresh diversifying bet) and by the dashboard to render a clustermap so
the operator can see the structure at a glance.

All math is intentionally pure-Python — no numpy import, no scipy. The
inputs (60 daily log-returns) are tiny; vectorization buys nothing here
and pure Python keeps the dependency surface clean.

Returns are computed as ``r_t = ln(close_t / close_{t-1})``. We accept
the close-price series and convert internally so callers don't have to
remember whether they need prices or returns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Minimum overlapping return-points required to trust a correlation —
# below this the sample variance is too noisy to be meaningful.
_MIN_OVERLAP = 30

# Penalty thresholds — hard-coded for now; tune empirically once round-trips
# accumulate (Phase 5 ML layer can sweep them later).
_CORR_HIGH = 0.7
_CORR_MED = 0.5
_PENALTY_HIGH = -0.5
_PENALTY_MED = -0.25


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """Symmetric correlation matrix between symbols.

    Stored as a flat ``{(sym_a, sym_b): rho}`` dict keyed by sorted-tuple
    so callers don't need to know which order the pair was added in.
    Diagonal entries (sym, sym) are always 1.0.

    Pairs that lacked enough overlap (``< _MIN_OVERLAP`` shared bars) are
    omitted from ``pairs`` and reported in ``insufficient_pairs`` for the
    dashboard / journal to surface honestly rather than fake a 0.0.
    """

    symbols: list[str]
    pairs: dict[tuple[str, str], float]
    insufficient_pairs: list[tuple[str, str]]

    def get(self, a: str, b: str) -> float | None:
        """Return ρ(a, b), or ``None`` if the pair was undercomputed."""
        if a == b:
            return 1.0
        key = _key(a, b)
        return self.pairs.get(key)

    def to_journal_payload(self) -> dict[str, object]:
        """JSON-friendly form for the trade journal / dashboard."""
        return {
            "symbols": list(self.symbols),
            "pairs": [
                {"a": a, "b": b, "rho": round(rho, 4)}
                for (a, b), rho in self.pairs.items()
            ],
            "insufficient_pairs": [
                {"a": a, "b": b} for (a, b) in self.insufficient_pairs
            ],
        }


def closes_to_log_returns(closes: Sequence[float]) -> list[float]:
    """Compute the log-return vector from a close-price series.

    Skips any zero / negative / non-finite price (degenerate data) by
    returning a NaN-free run of valid points only — the caller's job is
    to align timestamps if it cares about strict time-matching, but for
    Pearson on equal-length series produced from the same daily-bar
    pipeline, that's already guaranteed.
    """
    out: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0 or cur <= 0:
            continue
        try:
            out.append(math.log(cur / prev))
        except (ValueError, OverflowError):
            continue
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Standard Pearson correlation. Returns ``None`` when undefined.

    Undefined cases:
      * lengths mismatch or below ``_MIN_OVERLAP`` after truncating to
        the shorter length;
      * either series has zero variance (constant input → divide-by-zero).
    """
    n = min(len(x), len(y))
    if n < _MIN_OVERLAP:
        return None
    xs = list(x[-n:])
    ys = list(y[-n:])
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for xi, yi in zip(xs, ys, strict=True):
        dx = xi - mean_x
        dy = yi - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def compute_correlation_matrix(
    returns_by_symbol: Mapping[str, Sequence[float]],
) -> CorrelationMatrix:
    """Build the full pairwise correlation matrix between every symbol.

    Pairs whose overlap is below ``_MIN_OVERLAP`` are recorded in
    ``insufficient_pairs`` instead of given a fake 0.0 — the dashboard
    can surface them as "—" and the engine can treat them as no signal.
    """
    symbols = sorted(returns_by_symbol)
    pairs: dict[tuple[str, str], float] = {}
    insufficient: list[tuple[str, str]] = []
    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            rho = pearson(returns_by_symbol[a], returns_by_symbol[b])
            if rho is None:
                insufficient.append(_key(a, b))
            else:
                pairs[_key(a, b)] = rho
    return CorrelationMatrix(
        symbols=symbols, pairs=pairs, insufficient_pairs=insufficient
    )


def correlation_penalty_for(
    matrix: CorrelationMatrix,
    candidate: str,
    held_positions: Sequence[str],
) -> float:
    """Return the candidate's penalty given current holdings.

    The penalty fires off the *largest absolute* correlation between the
    candidate and any held position — both +0.85 and −0.85 represent the
    same kind of stacked exposure (just opposite-side). Returning 0.0
    when the candidate is the held position itself prevents
    self-stacking (the position-fit bonus already handles that case).
    """
    if not held_positions:
        return 0.0
    max_abs = 0.0
    for h in held_positions:
        if h == candidate:
            continue
        rho = matrix.get(candidate, h)
        if rho is None:
            continue
        max_abs = max(max_abs, abs(rho))
    if max_abs > _CORR_HIGH:
        return _PENALTY_HIGH
    if max_abs > _CORR_MED:
        return _PENALTY_MED
    return 0.0


def _key(a: str, b: str) -> tuple[str, str]:
    """Order-independent key for ``pairs`` dict lookups."""
    return (a, b) if a < b else (b, a)


__all__ = [
    "CorrelationMatrix",
    "closes_to_log_returns",
    "compute_correlation_matrix",
    "correlation_penalty_for",
    "pearson",
]
