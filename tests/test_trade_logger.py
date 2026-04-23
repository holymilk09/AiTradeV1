"""Trade journal writes both JSONL and SQLite mirror."""

from __future__ import annotations

import json
from pathlib import Path

from aitrade.execution.orders import OrderRequest, Side
from aitrade.logging.trade_logger import EventType, TradeLogger


def test_writes_jsonl_and_queryable_sqlite(tmp_path: Path) -> None:
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10, strategy_id="test")
        journal.log_submit(order)
        journal.log_risk_block(order, reason="test block")

        events = journal.events_for_run()
        types = [e["event_type"] for e in events]
        assert EventType.ORDER_SUBMITTED.value in types
        assert EventType.RISK_BLOCKED.value in types

    jsonl = (tmp_path / "trades.jsonl").read_text().strip().splitlines()
    assert len(jsonl) >= 3  # run_start + submit + block + run_end
    parsed = [json.loads(line) for line in jsonl]
    assert any(e["event_type"] == EventType.RISK_BLOCKED.value for e in parsed)
