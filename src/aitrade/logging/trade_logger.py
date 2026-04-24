"""Structured trade journal — JSONL (append-only) + SQLite (queryable).

This is the substrate for post-trade learning. Every order, fill, and cancel
is written here. Queries run against the SQLite mirror; the JSONL is the
ground-truth audit trail.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from aitrade.execution.orders import Fill, OrderAck, OrderRequest


class EventType(str, Enum):
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACK = "order_ack"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"
    RISK_BLOCKED = "risk_blocked"
    REASONER_DECISION = "reasoner_decision"
    DISCOVERY_SCAN = "discovery_scan"
    PATTERN_DETECTED = "pattern_detected"
    CANDIDATE_BOARD = "candidate_board"
    MARKET_SNAPSHOT = "market_snapshot"
    FLOOR_TRADER_DECISION = "floor_trader_decision"
    ROUND_TRIP_CLOSED = "round_trip_closed"
    STALE_DATA_SKIP = "stale_data_skip"
    EMPTY_DISCOVERY = "empty_discovery"
    RUN_START = "run_start"
    RUN_END = "run_end"


@dataclass(frozen=True, slots=True)
class TradeEvent:
    event_id: str
    run_id: str
    strategy_id: str
    event_type: EventType
    symbol: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_events (
    event_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    strategy_id  TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trade_events_run ON trade_events(run_id);
CREATE INDEX IF NOT EXISTS idx_trade_events_symbol ON trade_events(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_events_type ON trade_events(event_type);
"""


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


class TradeLogger:
    def __init__(
        self,
        *,
        log_dir: Path,
        run_id: str | None = None,
        strategy_id: str = "unknown",
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        self.run_id = run_id or f"run-{stamp}-{uuid4().hex[:6]}"
        self.strategy_id = strategy_id
        self._jsonl_path = self.log_dir / "trades.jsonl"
        self._db_path = self.log_dir / "trades.sqlite"
        self._conn = sqlite3.connect(self._db_path, isolation_level=None)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TradeLogger:
        self.record(EventType.RUN_START, symbol="-", payload={"started_at": datetime.now(UTC)})
        return self

    def __exit__(self, *exc: object) -> None:
        self.record(EventType.RUN_END, symbol="-", payload={"ended_at": datetime.now(UTC)})
        self.close()

    def record(
        self,
        event_type: EventType,
        *,
        symbol: str,
        payload: dict[str, Any] | None = None,
    ) -> TradeEvent:
        event = TradeEvent(
            event_id=uuid4().hex,
            run_id=self.run_id,
            strategy_id=self.strategy_id,
            event_type=event_type,
            symbol=symbol,
            timestamp=datetime.now(UTC),
            payload=payload or {},
        )
        line = json.dumps(
            {
                "event_id": event.event_id,
                "run_id": event.run_id,
                "strategy_id": event.strategy_id,
                "event_type": event.event_type.value,
                "symbol": event.symbol,
                "timestamp": event.timestamp.isoformat(),
                "payload": event.payload,
            },
            default=_json_default,
        )
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        self._conn.execute(
            "INSERT INTO trade_events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.run_id,
                event.strategy_id,
                event.event_type.value,
                event.symbol,
                event.timestamp.isoformat(),
                json.dumps(event.payload, default=_json_default),
            ),
        )
        return event

    def log_submit(self, order: OrderRequest) -> None:
        self.record(EventType.ORDER_SUBMITTED, symbol=order.symbol, payload={"order": order})

    def log_ack(self, order: OrderRequest, ack: OrderAck) -> None:
        self.record(
            EventType.ORDER_ACK,
            symbol=order.symbol,
            payload={"order": order, "ack": ack},
        )

    def log_fill(self, fill: Fill) -> None:
        self.record(EventType.ORDER_FILLED, symbol=fill.symbol, payload={"fill": fill})

    def log_risk_block(self, order: OrderRequest, reason: str) -> None:
        self.record(
            EventType.RISK_BLOCKED,
            symbol=order.symbol,
            payload={"order": order, "reason": reason},
        )

    def events_for_run(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rid = run_id or self.run_id
        cur = self._conn.execute(
            "SELECT event_id, run_id, strategy_id, event_type, symbol, timestamp, payload "
            "FROM trade_events WHERE run_id = ? ORDER BY timestamp",
            (rid,),
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
