"""SQL-backed read-only views over the trade journal.

These helpers power the CLI: filter round-trips by pattern or symbol,
roll up win-rate per pattern, and emit a Markdown digest suitable for
pasting into a fresh Claude context.

Everything here is read-only; mutations go through
:class:`aitrade.journal.round_trips.RoundTripReconciler`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aitrade.journal.round_trips import (
    ExitReason,
    PnlBucket,
    TradeRoundTrip,
    _parse_ts,  # noqa: PLC2701 — reuse the journal's ISO timestamp parser
)
from aitrade.logging.trade_logger import TradeLogger

if TYPE_CHECKING:
    import sqlite3


# --- public dataclass --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatternStats:
    """Win-rate + P&L summary for a single pattern across all round-trips.

    A single round-trip with ``pattern_hits = ["volume_trend", "breakout"]``
    contributes one count to *each* of those patterns.
    """

    pattern: str
    n_trades: int
    n_wins: int
    n_losses: int
    n_breakeven: int
    avg_pnl_pct: float
    total_pnl_usd: float

    @property
    def win_rate(self) -> float:
        """Wins / total. Returns 0.0 when there are no trades."""
        if self.n_trades == 0:
            return 0.0
        return self.n_wins / self.n_trades


# --- main view layer ---------------------------------------------------------


class JournalViews:
    """SQL-backed read-only views over the trade journal."""

    def __init__(self, journal: TradeLogger) -> None:
        self._journal = journal
        self._conn: sqlite3.Connection = journal._conn  # noqa: SLF001 — same DB file

    # ----- filtered query ---------------------------------------------------

    def round_trips(
        self,
        *,
        symbol: str | None = None,
        pattern: str | None = None,
        wins_only: bool = False,
        losses_only: bool = False,
        limit: int = 100,
    ) -> list[TradeRoundTrip]:
        """Return matching round-trips, newest first, capped at ``limit``.

        ``pattern`` matches when the given name is a member of the row's
        ``pattern_hits`` JSON array — implemented via a quoted-substring
        LIKE because the array is stored as TEXT.

        ``wins_only`` / ``losses_only`` restrict by ``pnl_bucket``. They
        are mutually exclusive in spirit; if both are passed, the result
        is an empty list (a row can't be both).
        """
        if wins_only and losses_only:
            return []

        clauses: list[str] = []
        params: list[object] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if pattern is not None:
            # pattern_hits is a JSON array stored as TEXT, e.g.
            # ["volume_trend","breakout"]. Names are always wrapped in
            # double quotes inside, so the quoted-substring match is safe
            # against false positives like a pattern called "vol" matching
            # a row containing "volume_trend".
            clauses.append("pattern_hits LIKE ?")
            params.append(f'%"{pattern}"%')
        if wins_only:
            clauses.append("pnl_bucket = ?")
            params.append(PnlBucket.WIN.value)
        if losses_only:
            clauses.append("pnl_bucket = ?")
            params.append(PnlBucket.LOSS.value)

        sql = (
            "SELECT trade_id, symbol, strategy_id, entry_event_id, exit_event_id, "
            "entry_ts, exit_ts, entry_price, exit_price, qty, pnl_usd, pnl_pct, "
            "holding_secs, pnl_bucket, exit_reason, entry_thesis, entry_catalyst, "
            "entry_confidence, pattern_hits, market_snapshot "
            "FROM round_trips"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY exit_ts DESC LIMIT ?"
        params.append(int(limit))

        cur = self._conn.execute(sql, params)
        return [_row_to_round_trip(row) for row in cur.fetchall()]

    # ----- aggregate --------------------------------------------------------

    def pattern_stats(self) -> list[PatternStats]:
        """Win-rate + P&L summary per distinct pattern across all round-trips.

        One round-trip can contribute to multiple patterns. Sorted by
        trade count descending, with ties broken alphabetically so output
        is deterministic across runs.
        """
        cur = self._conn.execute(
            "SELECT pattern_hits, pnl_bucket, pnl_usd, pnl_pct FROM round_trips"
        )

        counts: dict[str, int] = defaultdict(int)
        wins: dict[str, int] = defaultdict(int)
        losses: dict[str, int] = defaultdict(int)
        breakeven: dict[str, int] = defaultdict(int)
        total_usd: dict[str, float] = defaultdict(float)
        total_pct: dict[str, float] = defaultdict(float)

        for row in cur.fetchall():
            patterns = _safe_pattern_hits(row[0])
            if not patterns:
                continue
            bucket = str(row[1])
            try:
                pnl_usd = float(row[2])
            except (TypeError, ValueError):
                pnl_usd = 0.0
            try:
                pnl_pct = float(row[3])
            except (TypeError, ValueError):
                pnl_pct = 0.0
            for name in patterns:
                counts[name] += 1
                total_usd[name] += pnl_usd
                total_pct[name] += pnl_pct
                if bucket == PnlBucket.WIN.value:
                    wins[name] += 1
                elif bucket == PnlBucket.LOSS.value:
                    losses[name] += 1
                else:
                    breakeven[name] += 1

        out: list[PatternStats] = []
        for name, n in counts.items():
            avg_pct = total_pct[name] / n if n else 0.0
            out.append(
                PatternStats(
                    pattern=name,
                    n_trades=n,
                    n_wins=wins[name],
                    n_losses=losses[name],
                    n_breakeven=breakeven[name],
                    avg_pnl_pct=avg_pct,
                    total_pnl_usd=total_usd[name],
                )
            )
        # Deterministic ordering: most-traded first, alphabetical tiebreak.
        out.sort(key=lambda s: (-s.n_trades, s.pattern))
        return out

    # ----- LLM digest -------------------------------------------------------

    def export_for_llm(self, *, max_trades: int = 200) -> str:
        """Markdown digest of the most recent round-trips for LLM context.

        Joins ``trade_narratives`` when that table exists; falls back to
        a structured-fact dump otherwise so this works on a fresh journal
        before narratives have been built.
        """
        narratives = self._narratives_by_trade_id() if self._has_narratives_table() else {}

        cur = self._conn.execute(
            "SELECT trade_id, symbol, strategy_id, entry_event_id, exit_event_id, "
            "entry_ts, exit_ts, entry_price, exit_price, qty, pnl_usd, pnl_pct, "
            "holding_secs, pnl_bucket, exit_reason, entry_thesis, entry_catalyst, "
            "entry_confidence, pattern_hits, market_snapshot "
            "FROM round_trips ORDER BY exit_ts DESC LIMIT ?",
            (int(max_trades),),
        )
        rows = cur.fetchall()

        lines: list[str] = []
        lines.append(f"# Trade journal — last {len(rows)} round-trips")
        lines.append("")
        if not rows:
            lines.append("_No round-trips recorded yet._")
            return "\n".join(lines) + "\n"

        for idx, row in enumerate(rows, start=1):
            rt = _row_to_round_trip(row)
            block = _format_trade_block(
                rt,
                index=idx,
                narrative=narratives.get(rt.trade_id),
            )
            lines.extend(block)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ----- internals --------------------------------------------------------

    def _has_narratives_table(self) -> bool:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'trade_narratives'"
        )
        return cur.fetchone() is not None

    def _narratives_by_trade_id(self) -> dict[str, str]:
        cur = self._conn.execute("SELECT trade_id, narrative FROM trade_narratives")
        return {str(row[0]): str(row[1]) for row in cur.fetchall()}


# --- module helpers ----------------------------------------------------------


def _safe_pattern_hits(raw: object) -> list[str]:
    """Best-effort decode of the pattern_hits column. Bad rows -> []."""
    if raw is None or not isinstance(raw, str) or not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(p) for p in parsed]


def _safe_market_snapshot(raw: object) -> dict[str, object] | None:
    if raw is None or not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(k): v for k, v in parsed.items()}


def _row_to_round_trip(row: tuple[Any, ...]) -> TradeRoundTrip:
    # SQLite columns are dynamically typed; we parse defensively below.
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
        pattern_hits=_safe_pattern_hits(row[18]),
        market_snapshot=_safe_market_snapshot(row[19]),
    )


def _format_holding(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    if secs < 3_600:
        return f"{secs / 60:.0f}m"
    if secs < 86_400:
        return f"{secs / 3_600:.1f}h"
    return f"{secs / 86_400:.1f} days"


def _bucket_sign(bucket: PnlBucket, pnl_pct: float) -> str:
    """Render the +/- prefix for the heading. Breakevens stay unsigned."""
    if bucket is PnlBucket.WIN:
        return f"+{pnl_pct * 100:.2f}%"
    if bucket is PnlBucket.LOSS:
        # pnl_pct is already negative; format keeps the sign.
        return f"{pnl_pct * 100:.2f}%"
    return f"{pnl_pct * 100:.2f}%"


def _format_trade_block(
    rt: TradeRoundTrip, *, index: int, narrative: str | None
) -> list[str]:
    """Render one H2 block. Mirrors the docstring spec in JournalViews."""
    pct = _bucket_sign(rt.pnl_bucket, rt.pnl_pct)
    heading = (
        f"## TRADE-{index:03d} — {rt.symbol} — {rt.pnl_bucket.value.upper()} {pct}"
    )

    lines: list[str] = [heading]
    lines.append(
        f"- Entry: {rt.entry_ts.strftime('%Y-%m-%d %H:%M')} @ ${rt.entry_price:.2f}"
    )
    lines.append(
        f"- Exit: {rt.exit_ts.strftime('%Y-%m-%d %H:%M')} @ ${rt.exit_price:.2f}"
    )
    lines.append(
        f"- Held {_format_holding(rt.holding_secs)}, "
        f"{rt.qty:g} share{'s' if rt.qty != 1 else ''}, "
        f"{rt.pnl_usd:+.2f} USD"
    )
    lines.append(
        f"- Bucket: {rt.pnl_bucket.value.upper()}, "
        f"exit reason: {rt.exit_reason.value.upper()}"
    )
    if rt.pattern_hits:
        lines.append(f"- Patterns: {', '.join(rt.pattern_hits)}")
    else:
        lines.append("- Patterns: (none)")

    regime = None
    if rt.market_snapshot is not None:
        raw = rt.market_snapshot.get("regime")
        if raw is not None:
            regime = str(raw)
    if regime is not None:
        lines.append(f"- Regime at entry: {regime}")

    if rt.entry_thesis:
        lines.append(f'- Thesis: "{rt.entry_thesis}"')
    if narrative:
        lines.append(f'- Narrative: "{narrative}"')

    return lines


__all__ = [
    "JournalViews",
    "PatternStats",
]
