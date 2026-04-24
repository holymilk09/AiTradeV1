"""Round-trip reconciler.

Pairs ORDER_FILLED events from the trade journal into closed BUY/SELL
:class:`TradeRoundTrip` rows using FIFO matching, and persists the open
inventory between reconciler runs so a session restart does not lose
state.

Each completed round-trip is tagged with two orthogonal labels:

* :class:`PnlBucket` — WIN / LOSS / BREAKEVEN (numeric outcome)
* :class:`ExitReason` — TARGET_HIT / STOP_HIT / MANUAL_EXIT /
  EOD_FLATTEN / RISK_HALT / UNKNOWN (causal outcome)

The reconciler is **idempotent**: replaying it over the same event log
must not produce duplicate rows.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger

from aitrade.execution.orders import Side
from aitrade.logging.trade_logger import EventType, TradeLogger

# --- public enums ------------------------------------------------------------


class PnlBucket(str, Enum):
    """Coarse PnL outcome, derived from ``pnl_pct``."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


class ExitReason(str, Enum):
    """Why the position closed. Sourced from the exit event payload when present."""

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    MANUAL_EXIT = "manual_exit"
    EOD_FLATTEN = "eod_flatten"
    RISK_HALT = "risk_halt"
    UNKNOWN = "unknown"


_BREAKEVEN_BAND = 0.001  # |pnl_pct| <= 0.1% counts as breakeven


# --- public dataclass --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeRoundTrip:
    """A closed buy → sell pair, fully matched, ready for analytics."""

    trade_id: str
    symbol: str
    strategy_id: str
    entry_event_id: str
    exit_event_id: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl_usd: float
    pnl_pct: float
    holding_secs: int
    pnl_bucket: PnlBucket
    exit_reason: ExitReason
    entry_thesis: str | None = None
    entry_catalyst: str | None = None
    entry_confidence: float | None = None
    pattern_hits: list[str] = field(default_factory=list)
    market_snapshot: dict[str, object] | None = None


# --- SQL ---------------------------------------------------------------------


_ROUND_TRIPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS round_trips (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy_id TEXT,
    entry_event_id TEXT NOT NULL,
    exit_event_id TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    exit_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    qty REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    holding_secs INTEGER NOT NULL,
    pnl_bucket TEXT NOT NULL,
    exit_reason TEXT NOT NULL,
    entry_thesis TEXT,
    entry_catalyst TEXT,
    entry_confidence REAL,
    pattern_hits TEXT,
    market_snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_rt_symbol ON round_trips(symbol);
CREATE INDEX IF NOT EXISTS idx_rt_pnl_bucket ON round_trips(pnl_bucket);
CREATE INDEX IF NOT EXISTS idx_rt_strategy ON round_trips(strategy_id);
"""

_OPEN_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS open_positions (
    lot_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy_id TEXT,
    entry_event_id TEXT NOT NULL,
    entry_ts TEXT NOT NULL,
    entry_price REAL NOT NULL,
    qty_remaining REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_op_symbol ON open_positions(symbol);
"""

_PROCESSED_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS reconciled_events (
    event_id TEXT PRIMARY KEY
);
"""


# --- internal helpers --------------------------------------------------------


def _bucket_for(pnl_pct: float) -> PnlBucket:
    if pnl_pct > _BREAKEVEN_BAND:
        return PnlBucket.WIN
    if pnl_pct < -_BREAKEVEN_BAND:
        return PnlBucket.LOSS
    return PnlBucket.BREAKEVEN


def _exit_reason_from(payload: dict[str, Any]) -> ExitReason:
    """Pluck an exit reason from an exit-event payload, defaulting to UNKNOWN."""
    raw = payload.get("exit_reason")
    if raw is None:
        # Many call sites stash details under nested dicts (e.g. ``fill``).
        for k in ("fill", "order"):
            sub = payload.get(k)
            if isinstance(sub, dict):
                raw = sub.get("exit_reason")
                if raw is not None:
                    break
    if raw is None:
        return ExitReason.UNKNOWN
    if isinstance(raw, ExitReason):
        return raw
    try:
        return ExitReason(str(raw))
    except ValueError:
        return ExitReason.UNKNOWN


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


@dataclass
class _OpenLot:
    """Mutable in-memory lot. Persisted between calls via ``open_positions``."""

    lot_id: str
    symbol: str
    strategy_id: str
    entry_event_id: str
    entry_ts: datetime
    entry_price: float
    qty_remaining: float


def _fill_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Trade events nest the Fill DTO under ``fill``. Tolerate flat shapes too."""
    nested = payload.get("fill")
    if isinstance(nested, dict):
        return nested
    return payload


def _fill_qty(payload: dict[str, Any]) -> float:
    fill = _fill_payload(payload)
    qty = fill.get("qty", payload.get("qty", 0.0))
    return float(qty)


def _fill_price(payload: dict[str, Any]) -> float:
    fill = _fill_payload(payload)
    price = fill.get("price", payload.get("price", 0.0))
    return float(price)


def _fill_side(payload: dict[str, Any]) -> Side | None:
    fill = _fill_payload(payload)
    raw = fill.get("side", payload.get("side"))
    if raw is None:
        return None
    if isinstance(raw, Side):
        return raw
    try:
        return Side(str(raw))
    except ValueError:
        return None


# --- main reconciler ---------------------------------------------------------


class RoundTripReconciler:
    """FIFO buy/sell pairer that persists round-trips and open lots to SQLite.

    The reconciler reads ORDER_FILLED events from the journal's SQLite mirror
    (so it stays consistent with replays) and walks them in chronological
    order. For each symbol it maintains a FIFO queue of open BUY lots; SELL
    fills consume those lots in arrival order, emitting a ``TradeRoundTrip``
    for each match.

    Calling :meth:`reconcile` multiple times on the same data is safe — each
    event is only consumed once thanks to the ``reconciled_events`` table.
    """

    def __init__(self, journal: TradeLogger) -> None:
        self._journal = journal
        self._conn = journal._conn  # noqa: SLF001 — same DB file, single connection
        self._conn.executescript(_ROUND_TRIPS_SCHEMA)
        self._conn.executescript(_OPEN_POSITIONS_SCHEMA)
        self._conn.executescript(_PROCESSED_EVENTS_SCHEMA)

    # ----- public API -------------------------------------------------------

    def reconcile(self, *, run_id: str | None = None) -> list[TradeRoundTrip]:
        """Pair unmatched ORDER_FILLED events into round trips.

        Args:
            run_id: Restrict to one run. ``None`` means all runs.

        Returns:
            Newly-created round trips (does not include ones inserted by
            previous calls).
        """
        events = self._unprocessed_fill_events(run_id=run_id)
        if not events:
            return []

        lots_by_symbol: dict[str, list[_OpenLot]] = self._load_open_lots()
        new_round_trips: list[TradeRoundTrip] = []
        consumed_event_ids: list[str] = []

        for event in events:
            symbol = event["symbol"]
            payload = event["payload"]
            side = _fill_side(payload)
            qty = _fill_qty(payload)
            price = _fill_price(payload)
            if side is None or qty <= 0 or price <= 0:
                # Malformed fill — skip but mark consumed so we don't loop.
                consumed_event_ids.append(event["event_id"])
                continue

            ts = _parse_ts(event["timestamp"])

            if side is Side.BUY:
                lot = _OpenLot(
                    lot_id=uuid4().hex,
                    symbol=symbol,
                    strategy_id=event["strategy_id"],
                    entry_event_id=event["event_id"],
                    entry_ts=ts,
                    entry_price=price,
                    qty_remaining=qty,
                )
                lots_by_symbol.setdefault(symbol, []).append(lot)
                consumed_event_ids.append(event["event_id"])
                continue

            # SELL: drain FIFO lots
            queue = lots_by_symbol.get(symbol, [])
            sell_remaining = qty
            while sell_remaining > 0 and queue:
                lot = queue[0]
                take = min(lot.qty_remaining, sell_remaining)
                pnl_usd = (price - lot.entry_price) * take
                pnl_pct = (price - lot.entry_price) / lot.entry_price
                holding_secs = max(int((ts - lot.entry_ts).total_seconds()), 0)
                rt = TradeRoundTrip(
                    trade_id=uuid4().hex,
                    symbol=symbol,
                    strategy_id=event["strategy_id"],
                    entry_event_id=lot.entry_event_id,
                    exit_event_id=event["event_id"],
                    entry_ts=lot.entry_ts,
                    exit_ts=ts,
                    entry_price=lot.entry_price,
                    exit_price=price,
                    qty=take,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    holding_secs=holding_secs,
                    pnl_bucket=_bucket_for(pnl_pct),
                    exit_reason=_exit_reason_from(payload),
                )
                new_round_trips.append(rt)
                lot.qty_remaining -= take
                sell_remaining -= take
                if lot.qty_remaining <= 0:
                    queue.pop(0)
            if sell_remaining > 0:
                # Sold more than we had on the books — likely a short or a
                # bookkeeping gap. Phase 1 doesn't model shorts; log and drop.
                logger.warning(
                    "reconciler: SELL of {} {} exceeds open lots by {:.4f}; ignoring excess",
                    symbol,
                    qty,
                    sell_remaining,
                )
            consumed_event_ids.append(event["event_id"])

        self._persist(
            new_round_trips=new_round_trips,
            lots_by_symbol=lots_by_symbol,
            consumed_event_ids=consumed_event_ids,
        )
        return new_round_trips

    def all_round_trips(self, *, symbol: str | None = None) -> list[TradeRoundTrip]:
        """Return every persisted round trip, newest last. Optionally filter by symbol."""
        if symbol is None:
            cur = self._conn.execute(
                "SELECT trade_id, symbol, strategy_id, entry_event_id, exit_event_id, "
                "entry_ts, exit_ts, entry_price, exit_price, qty, pnl_usd, pnl_pct, "
                "holding_secs, pnl_bucket, exit_reason, entry_thesis, entry_catalyst, "
                "entry_confidence, pattern_hits, market_snapshot "
                "FROM round_trips ORDER BY exit_ts"
            )
        else:
            cur = self._conn.execute(
                "SELECT trade_id, symbol, strategy_id, entry_event_id, exit_event_id, "
                "entry_ts, exit_ts, entry_price, exit_price, qty, pnl_usd, pnl_pct, "
                "holding_secs, pnl_bucket, exit_reason, entry_thesis, entry_catalyst, "
                "entry_confidence, pattern_hits, market_snapshot "
                "FROM round_trips WHERE symbol = ? ORDER BY exit_ts",
                (symbol,),
            )
        out: list[TradeRoundTrip] = []
        for row in cur.fetchall():
            out.append(_row_to_round_trip(row))
        return out

    # ----- internals --------------------------------------------------------

    def _unprocessed_fill_events(self, *, run_id: str | None) -> list[dict[str, Any]]:
        if run_id is None:
            cur = self._conn.execute(
                "SELECT event_id, run_id, strategy_id, event_type, symbol, timestamp, payload "
                "FROM trade_events "
                "WHERE event_type = ? "
                "AND event_id NOT IN (SELECT event_id FROM reconciled_events) "
                "ORDER BY timestamp, event_id",
                (EventType.ORDER_FILLED.value,),
            )
        else:
            cur = self._conn.execute(
                "SELECT event_id, run_id, strategy_id, event_type, symbol, timestamp, payload "
                "FROM trade_events "
                "WHERE event_type = ? AND run_id = ? "
                "AND event_id NOT IN (SELECT event_id FROM reconciled_events) "
                "ORDER BY timestamp, event_id",
                (EventType.ORDER_FILLED.value, run_id),
            )
        out: list[dict[str, Any]] = []
        for row in cur.fetchall():
            out.append(
                {
                    "event_id": row[0],
                    "run_id": row[1],
                    "strategy_id": row[2],
                    "event_type": row[3],
                    "symbol": row[4],
                    "timestamp": row[5],
                    "payload": json.loads(row[6]),
                }
            )
        return out

    def _load_open_lots(self) -> dict[str, list[_OpenLot]]:
        cur = self._conn.execute(
            "SELECT lot_id, symbol, strategy_id, entry_event_id, entry_ts, "
            "entry_price, qty_remaining "
            "FROM open_positions ORDER BY entry_ts, lot_id"
        )
        out: dict[str, list[_OpenLot]] = {}
        for row in cur.fetchall():
            lot = _OpenLot(
                lot_id=row[0],
                symbol=row[1],
                strategy_id=row[2] or "",
                entry_event_id=row[3],
                entry_ts=_parse_ts(row[4]),
                entry_price=float(row[5]),
                qty_remaining=float(row[6]),
            )
            out.setdefault(lot.symbol, []).append(lot)
        return out

    def _persist(
        self,
        *,
        new_round_trips: list[TradeRoundTrip],
        lots_by_symbol: dict[str, list[_OpenLot]],
        consumed_event_ids: list[str],
    ) -> None:
        # Single transaction so a crash mid-way does not double-insert next call.
        try:
            self._conn.execute("BEGIN")
            for rt in new_round_trips:
                self._conn.execute(
                    "INSERT OR IGNORE INTO round_trips VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rt.trade_id,
                        rt.symbol,
                        rt.strategy_id,
                        rt.entry_event_id,
                        rt.exit_event_id,
                        rt.entry_ts.isoformat(),
                        rt.exit_ts.isoformat(),
                        rt.entry_price,
                        rt.exit_price,
                        rt.qty,
                        rt.pnl_usd,
                        rt.pnl_pct,
                        rt.holding_secs,
                        rt.pnl_bucket.value,
                        rt.exit_reason.value,
                        rt.entry_thesis,
                        rt.entry_catalyst,
                        rt.entry_confidence,
                        json.dumps(rt.pattern_hits),
                        json.dumps(rt.market_snapshot) if rt.market_snapshot else None,
                    ),
                )

            # Replace open_positions snapshot wholesale: it represents *current* state.
            self._conn.execute("DELETE FROM open_positions")
            for queue in lots_by_symbol.values():
                for lot in queue:
                    if lot.qty_remaining <= 0:
                        continue
                    self._conn.execute(
                        "INSERT INTO open_positions VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            lot.lot_id,
                            lot.symbol,
                            lot.strategy_id,
                            lot.entry_event_id,
                            lot.entry_ts.isoformat(),
                            lot.entry_price,
                            lot.qty_remaining,
                        ),
                    )

            for eid in consumed_event_ids:
                self._conn.execute(
                    "INSERT OR IGNORE INTO reconciled_events VALUES (?)", (eid,)
                )
            self._conn.execute("COMMIT")
        except sqlite3.Error:
            self._conn.execute("ROLLBACK")
            raise


def _row_to_round_trip(row: tuple[Any, ...]) -> TradeRoundTrip:
    return TradeRoundTrip(
        trade_id=row[0],
        symbol=row[1],
        strategy_id=row[2] or "",
        entry_event_id=row[3],
        exit_event_id=row[4],
        entry_ts=_parse_ts(row[5]),
        exit_ts=_parse_ts(row[6]),
        entry_price=float(row[7]),
        exit_price=float(row[8]),
        qty=float(row[9]),
        pnl_usd=float(row[10]),
        pnl_pct=float(row[11]),
        holding_secs=int(row[12]),
        pnl_bucket=PnlBucket(row[13]),
        exit_reason=ExitReason(row[14]),
        entry_thesis=row[15],
        entry_catalyst=row[16],
        entry_confidence=row[17],
        pattern_hits=json.loads(row[18]) if row[18] else [],
        market_snapshot=json.loads(row[19]) if row[19] else None,
    )


__all__ = [
    "ExitReason",
    "PnlBucket",
    "RoundTripReconciler",
    "TradeRoundTrip",
]
