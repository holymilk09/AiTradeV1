"""Postgres mirror for TradeLogger.

The existing `aitrade.logging.trade_logger.TradeLogger` writes to JSONL
(append-only ground truth) + SQLite (queryable). This module adds an
optional Postgres sink for the dashboard. JSONL remains authoritative; if
the Postgres write fails the JSONL line is still written and the failure is
logged, never raised — we never lose journal data because the dashboard DB
is down.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import text

from aitrade.api.db import session_scope

log = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


async def mirror_event(
    *,
    event_id: str,
    run_id: str,
    strategy_id: str | None,
    event_type: str,
    ticker: str | None,
    timestamp: datetime,
    payload: dict[str, Any],
    account: str | None = None,
    severity: str = "info",
    source: str = "auto",
) -> None:
    payload_json = json.dumps(payload, default=_json_safe)
    try:
        async with session_scope() as s:
            await s.execute(
                text(
                    """
                    INSERT INTO journal_events
                      (event_id, ts, run_id, strategy_id, account, ticker,
                       event_type, severity, payload, source)
                    VALUES
                      (:event_id, :ts, :run_id, :strategy_id, :account, :ticker,
                       :event_type, :severity, CAST(:payload AS jsonb), :source)
                    ON CONFLICT (event_id) DO NOTHING
                    """
                ),
                {
                    "event_id": event_id,
                    "ts": timestamp,
                    "run_id": run_id,
                    "strategy_id": strategy_id,
                    "account": account,
                    "ticker": ticker,
                    "event_type": event_type,
                    "severity": severity,
                    "payload": payload_json,
                    "source": source,
                },
            )
            await s.commit()
    except Exception as e:  # noqa: BLE001
        # Never raise — JSONL is authoritative.
        log.warning("journal mirror failed event_id=%s err=%s", event_id, e)
