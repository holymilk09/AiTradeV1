"""Candidate board: combine buzz + pattern hits into a ranked decision board.

The engine each cycle:
  1. Discovery → ``DiscoveredTicker`` per symbol
  2. Pattern scan → ``list[PatternSignal]`` per symbol
  3. Build → ``CandidateBoard`` (this module)
  4. Reasoner picks one (or passes)

Scoring (from the plan, pinned in the self-review):

    pattern_score          = sum(s.score for s in signals)
    combined_score         = z(buzz_score) + z(pattern_score) + 0.5 * position_fit_bonus

Z-scoring is done **within the cycle's candidates** so a quiet day's top buzz
score isn't unfairly punished against a busy day's. ``position_fit_bonus``
favors candidates the account can actually take a fresh position in:

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


@dataclass(frozen=True, slots=True)
class Candidate:
    """One row on the board — everything the reasoner needs to compare picks."""

    symbol: str
    buzz_score: float
    pattern_score: float
    combined_score: float
    position_fit_bonus: float
    pattern_hits: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dump for logging + reasoner prompt."""
        return {
            "symbol": self.symbol,
            "buzz_score": round(self.buzz_score, 4),
            "pattern_score": round(self.pattern_score, 4),
            "combined_score": round(self.combined_score, 4),
            "position_fit_bonus": round(self.position_fit_bonus, 4),
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
    held_qty_by_symbol: dict[str, float] | None = None,
    last_price_by_symbol: dict[str, float] | None = None,
    cash_available: float = 0.0,
    target_notional: float = 0.0,
    now: datetime | None = None,
) -> CandidateBoard:
    """Combine discovery + pattern scans into a sorted ``CandidateBoard``.

    Symbols with no buzz AND no patterns are dropped — they wouldn't even be
    candidates. Per-cycle z-scoring keeps scores comparable across days with
    different baseline buzz volumes.
    """
    held_qty_by_symbol = held_qty_by_symbol or {}
    last_price_by_symbol = last_price_by_symbol or {}
    now = now or datetime.now(UTC)

    # Union of every symbol that surfaced in either discovery or patterns.
    by_buzz: dict[str, DiscoveredTicker] = {d.symbol: d for d in discovered}
    symbols = sorted(set(by_buzz) | set(patterns_by_symbol))
    if not symbols:
        return CandidateBoard(candidates=[], built_at=now)

    buzz_raw: list[float] = [by_buzz[s].buzz_score if s in by_buzz else 0.0 for s in symbols]
    pat_raw: list[float] = [
        sum(p.score for p in patterns_by_symbol.get(s, [])) for s in symbols
    ]

    buzz_z = _z_scores(buzz_raw)
    pat_z = _z_scores(pat_raw)

    rows: list[Candidate] = []
    for i, sym in enumerate(symbols):
        buzz = buzz_raw[i]
        pat = pat_raw[i]
        # Skip pure noise — symbols with no buzz AND no pattern firing.
        if buzz == 0.0 and pat == 0.0:
            continue
        signals = patterns_by_symbol.get(sym, [])
        fit = _position_fit_bonus(
            sym,
            held_qty=held_qty_by_symbol.get(sym, 0.0),
            reference_price=last_price_by_symbol.get(sym, 0.0),
            cash_available=cash_available,
            target_notional=target_notional,
        )
        combined = buzz_z[i] + pat_z[i] + 0.5 * fit
        evidence: dict[str, Any] = {
            "buzz_z": round(buzz_z[i], 4),
            "pattern_z": round(pat_z[i], 4),
            "raw_buzz": round(buzz, 4),
            "raw_pattern": round(pat, 4),
        }
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

        rows.append(
            Candidate(
                symbol=sym,
                buzz_score=buzz,
                pattern_score=pat,
                combined_score=combined,
                position_fit_bonus=fit,
                pattern_hits=[s.name for s in signals],
                evidence=evidence,
                discovered_at=by_buzz[sym].discovered_at if sym in by_buzz else None,
            )
        )

    rows.sort(key=lambda c: c.combined_score, reverse=True)
    return CandidateBoard(candidates=rows, built_at=now)
