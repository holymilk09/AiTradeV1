"""Similarity retrieval for past round-trips.

Given a candidate (symbol + pattern hits + market regime), pick the top-K
most similar already-closed round trips from the journal. The result is a
small list of :class:`TradeRoundTrip` rows that the LLM reasoner can fold
into its prompt as "what we learned last time we saw something like this".

Phase 1.5 v1 keeps this dead simple: no embeddings, no vector store, no
LLM in the retrieval loop — just a hand-tuned weighted score over three
features (same symbol, shared pattern hits, same market regime). The
substrate is SQLite + plain Python.

Score weights are intentionally tentative; we'll tune them once we have
enough trade history to A/B them. Until then, the dominant signal is
"have we ever traded this exact name before".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from loguru import logger

from aitrade.journal.round_trips import (
    ExitReason,
    PnlBucket,
    TradeRoundTrip,
    _parse_ts,  # noqa: PLC2701 — reuse the journal's ISO timestamp parser
)
from aitrade.logging.trade_logger import TradeLogger


@dataclass(frozen=True, slots=True)
class SimilarityQuery:
    """The "what am I about to trade" side of the similarity lookup.

    ``market_regime`` should be the string value of a
    :class:`aitrade.market.regime.Regime` (e.g. ``"risk_on_low_vol"``) so
    it matches exactly the value stashed in ``round_trips.market_snapshot``.
    """

    symbol: str
    pattern_hits: list[str]
    market_regime: str | None


# Internal scoring row — a thin tuple of the columns we need plus the
# parsed JSON. We don't materialise full TradeRoundTrip objects until we
# know which rows we'll return, to keep the per-row cost down on big
# journals.
@dataclass(frozen=True, slots=True)
class _CandidateRow:
    # SQLite row tuples are dynamically typed; the column-by-column shape
    # is asserted at parse time by ``_row_to_round_trip``.
    row: tuple[Any, ...]
    pattern_hits: list[str]
    regime: str | None
    exit_ts_iso: str


class SimilarTradesFinder:
    """Score-based retrieval over closed round-trips.

    Weights (Phase 1.5 v1):

    * same symbol            → ``+SAME_SYMBOL_W``
    * shared pattern hit     → ``+SHARED_PATTERN_W`` per shared pattern,
      capped at ``SHARED_PATTERN_CAP`` shared patterns
    * same market regime     → ``+SAME_REGIME_W``

    Higher score = more similar. Ties broken by ``exit_ts`` descending so
    a recent trade beats a stale one.

    Retrieval only kicks in when at least ``min_matches`` round-trips
    exist globally — at t=0 the reasoner sees an empty list and the
    floor-trader prompt notes "no prior experience yet" rather than
    inventing a fake one.
    """

    SAME_SYMBOL_W: ClassVar[int] = 3
    SHARED_PATTERN_W: ClassVar[int] = 2
    SAME_REGIME_W: ClassVar[int] = 1
    SHARED_PATTERN_CAP: ClassVar[int] = 2  # max shared patterns counted

    def __init__(
        self,
        journal: TradeLogger,
        *,
        min_matches: int = 3,
    ) -> None:
        self._journal = journal
        self._conn = journal._conn  # noqa: SLF001 — same DB file, single connection
        self._min_matches = min_matches

    # ----- public API -------------------------------------------------------

    def find(self, query: SimilarityQuery, *, top_k: int = 3) -> list[TradeRoundTrip]:
        """Return up to ``top_k`` round-trips ordered by similarity desc.

        Returns ``[]`` when fewer than ``min_matches`` round-trips exist
        globally so we don't inject "experience" the bot doesn't have.
        """
        if top_k <= 0:
            return []

        total = self._count_round_trips()
        if total < self._min_matches:
            return []

        candidates = self._load_candidates()
        if not candidates:
            return []

        query_patterns = set(query.pattern_hits)
        scored: list[tuple[int, str, _CandidateRow]] = []
        for cand in candidates:
            score = self._score(cand, query=query, query_patterns=query_patterns)
            scored.append((score, cand.exit_ts_iso, cand))

        # Sort by (score DESC, exit_ts DESC). Python's sort is stable; sort
        # twice cheapest-key-first so the primary key (score) wins.
        scored.sort(key=lambda item: item[1], reverse=True)
        scored.sort(key=lambda item: item[0], reverse=True)

        chosen = scored[:top_k]
        return [_row_to_round_trip(item[2].row) for item in chosen]

    # ----- internals --------------------------------------------------------

    def _count_round_trips(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM round_trips")
        row = cur.fetchone()
        if row is None:
            return 0
        return int(row[0])

    def _load_candidates(self) -> list[_CandidateRow]:
        cur = self._conn.execute(
            "SELECT trade_id, symbol, strategy_id, entry_event_id, exit_event_id, "
            "entry_ts, exit_ts, entry_price, exit_price, qty, pnl_usd, pnl_pct, "
            "holding_secs, pnl_bucket, exit_reason, entry_thesis, entry_catalyst, "
            "entry_confidence, pattern_hits, market_snapshot "
            "FROM round_trips"
        )
        out: list[_CandidateRow] = []
        for row in cur.fetchall():
            patterns = _safe_load_pattern_hits(row[18], trade_id=row[0])
            if patterns is None:
                continue
            regime = _safe_load_regime(row[19], trade_id=row[0])
            # `exit_ts` lives at column index 6.
            exit_ts_iso = str(row[6])
            out.append(
                _CandidateRow(
                    row=row,
                    pattern_hits=patterns,
                    regime=regime,
                    exit_ts_iso=exit_ts_iso,
                )
            )
        return out

    def _score(
        self,
        cand: _CandidateRow,
        *,
        query: SimilarityQuery,
        query_patterns: set[str],
    ) -> int:
        score = 0
        if cand.row[1] == query.symbol:
            score += self.SAME_SYMBOL_W

        if query_patterns and cand.pattern_hits:
            shared = query_patterns.intersection(cand.pattern_hits)
            if shared:
                effective = min(len(shared), self.SHARED_PATTERN_CAP)
                score += self.SHARED_PATTERN_W * effective

        if (
            query.market_regime is not None
            and cand.regime is not None
            and cand.regime == query.market_regime
        ):
            score += self.SAME_REGIME_W

        return score


# --- module helpers ----------------------------------------------------------


def _safe_load_pattern_hits(raw: object, *, trade_id: object) -> list[str] | None:
    """Return the parsed pattern_hits list or ``None`` to drop the row.

    JSON corruption shouldn't poison retrieval; we just skip the row so a
    bad insert decades ago doesn't crash today's reasoner.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        logger.warning(
            "similarity: non-string pattern_hits trade_id={} type={}",
            trade_id,
            type(raw).__name__,
        )
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("similarity: malformed pattern_hits JSON trade_id={}", trade_id)
        return None
    if not isinstance(parsed, list):
        logger.warning(
            "similarity: pattern_hits not a list trade_id={} type={}",
            trade_id,
            type(parsed).__name__,
        )
        return None
    return [str(p) for p in parsed]


def _safe_load_regime(raw: object, *, trade_id: object) -> str | None:
    """Pull ``regime`` out of the market_snapshot JSON. Tolerate junk."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("similarity: malformed market_snapshot JSON trade_id={}", trade_id)
        return None
    if not isinstance(parsed, dict):
        return None
    regime = parsed.get("regime")
    if regime is None:
        return None
    return str(regime)


def _row_to_round_trip(row: tuple[Any, ...]) -> TradeRoundTrip:
    """Map a SELECT row back to a ``TradeRoundTrip``.

    Mirrors :func:`aitrade.journal.round_trips._row_to_round_trip` but
    duplicated locally to keep the dependency graph one-way (we already
    import the module above for types/parsers). Row types come from
    SQLite and are loose by construction — ``Any`` is appropriate here.
    """
    pattern_hits_raw = row[18]
    pattern_hits: list[str] = []
    if isinstance(pattern_hits_raw, str) and pattern_hits_raw:
        try:
            parsed = json.loads(pattern_hits_raw)
        except (json.JSONDecodeError, ValueError):
            parsed = []
        if isinstance(parsed, list):
            pattern_hits = [str(p) for p in parsed]

    market_snapshot_raw = row[19]
    market_snapshot: dict[str, object] | None = None
    if isinstance(market_snapshot_raw, str) and market_snapshot_raw:
        try:
            parsed_ms = json.loads(market_snapshot_raw)
        except (json.JSONDecodeError, ValueError):
            parsed_ms = None
        if isinstance(parsed_ms, dict):
            # Cast keys to str to keep the dataclass type honest.
            market_snapshot = {str(k): v for k, v in parsed_ms.items()}

    return TradeRoundTrip(
        trade_id=str(row[0]),
        symbol=str(row[1]),
        strategy_id=str(row[2] or ""),
        entry_event_id=str(row[3]),
        exit_event_id=str(row[4]),
        entry_ts=_parse_ts(str(row[5])),
        exit_ts=_parse_ts(str(row[6])),
        entry_price=float(row[7]),
        exit_price=float(row[8]),
        qty=float(row[9]),
        pnl_usd=float(row[10]),
        pnl_pct=float(row[11]),
        holding_secs=int(row[12]),
        pnl_bucket=PnlBucket(row[13]),
        exit_reason=ExitReason(row[14]),
        entry_thesis=row[15] if isinstance(row[15], str) else None,
        entry_catalyst=row[16] if isinstance(row[16], str) else None,
        entry_confidence=float(row[17]) if row[17] is not None else None,
        pattern_hits=pattern_hits,
        market_snapshot=market_snapshot,
    )


__all__ = [
    "SimilarTradesFinder",
    "SimilarityQuery",
]
