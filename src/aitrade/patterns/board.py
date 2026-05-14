"""Candidate board: combine buzz + pattern hits + trend strength into a board.

The engine each cycle:
  1. Discovery → ``DiscoveredTicker`` per symbol
  2. Pattern scan → ``list[PatternSignal]`` per symbol
  3. Trend scan → ``TrendScore`` per symbol (Phase 5)
  4. Build → ``CandidateBoard`` (this module)
  5. Reasoner picks one (or passes)

Scoring:

    pattern_score          = sum(s.score for s in signals)
    trend_score            = TrendHunter score in [0, 1]
    combined_score         = z(buzz) + z(pattern) + z(trend) + 0.5 * position_fit_bonus

Z-scoring is done **within the cycle's candidates** so a quiet day's top
component scores aren't unfairly punished against a busy day's. The trend
component is added with equal weight to buzz and pattern — clean trends are
just as actionable as buzz spikes, and surfacing them prevents the
floor-trader from chasing noisy breakouts in choppy tape.

``position_fit_bonus``:

    +1.0 if we hold no position in the symbol AND have cash for a full slot
    -1.0 if we already hold the symbol (would require scaling up — discouraged)
     0.0 otherwise

The board is the *ranking* fed to the LLM; the LLM still has the final pick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.patterns.base import PatternSignal
from aitrade.patterns.trend_hunter import TrendScore
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class Candidate:
    """One row on the board — everything the reasoner needs to compare picks."""

    symbol: str
    buzz_score: float
    pattern_score: float
    combined_score: float
    position_fit_bonus: float
    trend_score: float = 0.0
    trend_direction: Direction = Direction.FLAT
    correlation_penalty: float = 0.0
    pattern_hits: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dump for logging + reasoner prompt."""
        return {
            "symbol": self.symbol,
            "buzz_score": round(self.buzz_score, 4),
            "pattern_score": round(self.pattern_score, 4),
            "trend_score": round(self.trend_score, 4),
            "trend_direction": self.trend_direction.value,
            "combined_score": round(self.combined_score, 4),
            "position_fit_bonus": round(self.position_fit_bonus, 4),
            "correlation_penalty": round(self.correlation_penalty, 4),
            "pattern_hits": list(self.pattern_hits),
            "evidence": dict(self.evidence),
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
        }


@dataclass(frozen=True, slots=True)
class CandidateBoard:
    """Ranked snapshot of what's worth looking at this cycle.

    ``candidates`` is sorted desc by ``combined_score``; ``top(n)`` slices.
    ``built_at`` is the orchestrator timestamp — useful for staleness audits.
    """

    candidates: list[Candidate]
    built_at: datetime

    def top(self, n: int) -> list[Candidate]:
        """Top ``n`` candidates by combined score."""
        return self.candidates[:n]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dump for the journal + reasoner prompt."""
        return {
            "built_at": self.built_at.isoformat(),
            "count": len(self.candidates),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _z_scores(values: list[float]) -> list[float]:
    """Per-cycle z-score normalization. Constant series → all zeros (no info)."""
    if not values:
        return []
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def _position_fit_bonus(
    symbol: str,
    *,
    held_qty: float,
    reference_price: float,
    cash_available: float,
    target_notional: float,
) -> float:
    """Reward fresh entries with cash; penalize already-held names.

    Returns +1.0 when we have no position and the cash to open one,
    -1.0 when we already hold the symbol, 0.0 otherwise.
    """
    if held_qty > 0:
        return -1.0
    if reference_price <= 0:
        return 0.0
    needed = target_notional
    if cash_available >= needed:
        return 1.0
    return 0.0


def build_board(
    discovered: list[DiscoveredTicker],
    patterns_by_symbol: dict[str, list[PatternSignal]],
    *,
    trend_by_symbol: dict[str, TrendScore] | None = None,
    correlation_penalty_by_symbol: dict[str, float] | None = None,
    held_qty_by_symbol: dict[str, float] | None = None,
    last_price_by_symbol: dict[str, float] | None = None,
    cash_available: float = 0.0,
    target_notional: float = 0.0,
    now: datetime | None = None,
) -> CandidateBoard:
    """Combine discovery + pattern + trend scans into a sorted ``CandidateBoard``.

    Symbols with no buzz AND no patterns AND no trend signal are dropped.
    Per-cycle z-scoring keeps scores comparable across days with different
    baseline buzz / trend volumes.
    """
    held_qty_by_symbol = held_qty_by_symbol or {}
    last_price_by_symbol = last_price_by_symbol or {}
    trend_by_symbol = trend_by_symbol or {}
    correlation_penalty_by_symbol = correlation_penalty_by_symbol or {}
    now = now or datetime.now(UTC)

    # Union of every symbol that surfaced in any of the three lanes.
    by_buzz: dict[str, DiscoveredTicker] = {d.symbol: d for d in discovered}
    symbols = sorted(set(by_buzz) | set(patterns_by_symbol) | set(trend_by_symbol))
    if not symbols:
        return CandidateBoard(candidates=[], built_at=now)

    buzz_raw: list[float] = [by_buzz[s].buzz_score if s in by_buzz else 0.0 for s in symbols]
    pat_raw: list[float] = [
        sum(p.score for p in patterns_by_symbol.get(s, [])) for s in symbols
    ]
    trend_raw: list[float] = [
        trend_by_symbol[s].score if s in trend_by_symbol else 0.0 for s in symbols
    ]

    buzz_z = _z_scores(buzz_raw)
    pat_z = _z_scores(pat_raw)
    trend_z = _z_scores(trend_raw)

    rows: list[Candidate] = []
    for i, sym in enumerate(symbols):
        buzz = buzz_raw[i]
        pat = pat_raw[i]
        tr_score = trend_raw[i]
        # Skip pure noise — nothing on any lane.
        if buzz == 0.0 and pat == 0.0 and tr_score == 0.0:
            continue
        signals = patterns_by_symbol.get(sym, [])
        trend = trend_by_symbol.get(sym)
        fit = _position_fit_bonus(
            sym,
            held_qty=held_qty_by_symbol.get(sym, 0.0),
            reference_price=last_price_by_symbol.get(sym, 0.0),
            cash_available=cash_available,
            target_notional=target_notional,
        )
        corr_penalty = correlation_penalty_by_symbol.get(sym, 0.0)
        combined = buzz_z[i] + pat_z[i] + trend_z[i] + 0.5 * fit + corr_penalty
        evidence: dict[str, Any] = {
            "buzz_z": round(buzz_z[i], 4),
            "pattern_z": round(pat_z[i], 4),
            "trend_z": round(trend_z[i], 4),
            "raw_buzz": round(buzz, 4),
            "raw_pattern": round(pat, 4),
            "raw_trend": round(tr_score, 4),
        }
        if corr_penalty != 0.0:
            evidence["correlation_penalty"] = round(corr_penalty, 4)
        if sym in by_buzz:
            t = by_buzz[sym]
            evidence["mention_count"] = t.mention_count
            evidence["source_weight"] = t.source_weight
            if t.evidence:
                evidence["snippets"] = list(t.evidence)
        per_pattern = [
            {"name": s.name, "score": round(s.score, 4), "direction": s.direction.value}
            for s in signals
        ]
        if per_pattern:
            evidence["patterns"] = per_pattern
        if trend is not None and trend.score > 0:
            evidence["trend_components"] = {
                k: round(v, 4) for k, v in trend.components.items()
            }
            evidence["trend_is_strong"] = trend.is_strong

        rows.append(
            Candidate(
                symbol=sym,
                buzz_score=buzz,
                pattern_score=pat,
                combined_score=combined,
                position_fit_bonus=fit,
                trend_score=tr_score,
                trend_direction=trend.direction if trend is not None else Direction.FLAT,
                correlation_penalty=corr_penalty,
                pattern_hits=[s.name for s in signals],
                evidence=evidence,
                discovered_at=by_buzz[sym].discovered_at if sym in by_buzz else None,
            )
        )

    rows.sort(key=lambda c: c.combined_score, reverse=True)
    return CandidateBoard(candidates=rows, built_at=now)
