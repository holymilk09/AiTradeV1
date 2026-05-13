"""Weekly review: FIFO round-trip pairing + per-strategy stats."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Add scripts/ to path so the review module is importable in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import weekly_review as wr  # noqa: E402


def _event(ts: datetime, strategy: str, symbol: str, side: str, qty: float, px: float) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "strategy_id": strategy,
        "symbol": symbol,
        "event_type": "fill_received",
        "payload": {
            "fill": {"side": side, "filled_qty": qty, "avg_fill_price": px},
        },
    }


def test_round_trip_fifo_pairing(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    events = [
        _event(datetime(2025, 1, 6, tzinfo=UTC), "sma", "AAPL", "buy", 10, 100.0),
        _event(datetime(2025, 1, 7, tzinfo=UTC), "sma", "AAPL", "buy", 5, 110.0),
        _event(datetime(2025, 1, 8, tzinfo=UTC), "sma", "AAPL", "sell", 12, 120.0),
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))
    df = wr.load_events(log)
    trips = wr.round_trips(df)
    # 12 sell qty against 10@100 (full) + 2@110 (partial). Two trips.
    assert len(trips) == 2
    assert trips[0].pnl == (120 - 100) * 10
    assert trips[1].pnl == (120 - 110) * 2


def test_per_strategy_stats(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    events = [
        _event(datetime(2025, 1, 6, tzinfo=UTC), "sma", "AAPL", "buy", 10, 100.0),
        _event(datetime(2025, 1, 7, tzinfo=UTC), "sma", "AAPL", "sell", 10, 110.0),
        _event(datetime(2025, 1, 8, tzinfo=UTC), "boll", "MSFT", "buy", 5, 200.0),
        _event(datetime(2025, 1, 9, tzinfo=UTC), "boll", "MSFT", "sell", 5, 190.0),
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))
    rt = wr.summarize(wr.round_trips(wr.load_events(log)))
    stats = wr.per_strategy_stats(rt)
    assert set(stats.index) == {"sma", "boll"}
    assert stats.loc["sma", "trades"] == 1
    assert stats.loc["sma", "win_rate"] == 1.0
    assert stats.loc["boll", "win_rate"] == 0.0


def test_empty_log(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    log.write_text("")
    events = wr.load_events(log)
    assert events.empty
    assert wr.round_trips(events) == []


def test_weekly_expectancy(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    events = [
        _event(datetime(2025, 1, 6, tzinfo=UTC), "sma", "AAPL", "buy", 1, 100.0),
        _event(datetime(2025, 1, 8, tzinfo=UTC), "sma", "AAPL", "sell", 1, 105.0),
        _event(datetime(2025, 1, 14, tzinfo=UTC), "sma", "AAPL", "buy", 1, 110.0),
        _event(datetime(2025, 1, 16, tzinfo=UTC), "sma", "AAPL", "sell", 1, 108.0),
    ]
    log.write_text("\n".join(json.dumps(e) for e in events))
    rt = wr.summarize(wr.round_trips(wr.load_events(log)))
    weekly = wr.weekly_expectancy(rt)
    # Two weeks: 1 trade each.
    assert len(weekly) == 2
    assert {"strategy", "week", "trades", "expectancy", "pnl"}.issubset(weekly.columns)
    # Type sanity: pandas types vary so just compare numerically.
    assert weekly.iloc[0]["expectancy"] == 5.0
    assert weekly.iloc[1]["expectancy"] == -2.0


def test_handles_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    valid = _event(datetime(2025, 1, 6, tzinfo=UTC), "sma", "AAPL", "buy", 1, 100.0)
    log.write_text(json.dumps(valid) + "\n" + "{not json}\n")
    df = wr.load_events(log)
    assert len(df) == 1


def test_load_events_uses_since_filter(tmp_path: Path) -> None:
    log = tmp_path / "trades.jsonl"
    old = _event(datetime(2024, 12, 1, tzinfo=UTC), "sma", "AAPL", "buy", 1, 100.0)
    new = _event(datetime(2025, 1, 6, tzinfo=UTC), "sma", "AAPL", "buy", 1, 100.0)
    log.write_text("\n".join(json.dumps(e) for e in [old, new]))
    df = wr.load_events(log, since=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime())
    assert len(df) == 1
