"""Read-only snapshot helpers for the dashboard.

Each helper assembles one "card's worth" of data: account summary, open
positions, today's events, recent round-trips. They run inside a request
handler, so they should be fast and never block.

All helpers tolerate failure — a downstream API hiccup (e.g. Alpaca slow)
should degrade a card gracefully (return ``None`` or ``[]``) rather than
500 the whole dashboard.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from aitrade.brokers.base import BrokerClient
from aitrade.journal.round_trips import PnlBucket, RoundTripReconciler
from aitrade.logging.trade_logger import TradeLogger


@dataclass(frozen=True, slots=True)
class AccountCard:
    cash: float
    equity: float
    buying_power: float
    is_paper: bool
    currency: str


@dataclass(frozen=True, slots=True)
class PositionRow:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class JournalRow:
    entry_ts: datetime
    exit_ts: datetime
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_usd: float
    pnl_bucket: str
    exit_reason: str
    thesis: str | None


@dataclass(frozen=True, slots=True)
class TodayCard:
    n_decisions: int
    n_orders: int
    n_round_trips: int
    realized_pnl_today_usd: float


def fetch_account(broker: BrokerClient | None) -> AccountCard | None:
    """Top-of-page summary. Returns None on broker error or absence."""
    if broker is None:
        return None
    try:
        a = broker.get_account()
    except Exception as e:  # pragma: no cover — network noise
        logger.warning("dashboard: get_account failed: {}", e)
        return None
    return AccountCard(
        cash=a.cash,
        equity=a.equity,
        buying_power=a.buying_power,
        is_paper=a.is_paper,
        currency=a.currency,
    )


def fetch_positions(broker: BrokerClient | None) -> list[PositionRow]:
    """Open positions from the broker. Empty on error."""
    if broker is None:
        return []
    try:
        ps = broker.get_positions()
    except Exception as e:  # pragma: no cover
        logger.warning("dashboard: get_positions failed: {}", e)
        return []
    return [
        PositionRow(
            symbol=p.symbol,
            qty=p.qty,
            avg_entry_price=p.avg_entry_price,
            market_value=p.market_value,
            unrealized_pnl=p.unrealized_pnl,
        )
        for p in ps
    ]


def fetch_round_trips(
    log_dir: Path,
    *,
    limit: int = 20,
) -> list[JournalRow]:
    """Read the last N closed round-trips from the trade journal.

    We open a one-shot read-only sqlite handle so the dashboard never
    touches the journal that an active engine has open for writes.
    """
    db_path = log_dir / "trades.sqlite"
    if not db_path.exists():
        return []
    try:
        # Read-only URI prevents accidental writes from a stray query.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, isolation_level=None)
    except sqlite3.OperationalError:
        return []
    try:
        cur = conn.execute(
            """
            SELECT entry_ts, exit_ts, symbol, qty, entry_price, exit_price,
                   pnl_pct, pnl_usd, pnl_bucket, exit_reason, entry_thesis
              FROM round_trips
             ORDER BY exit_ts DESC
             LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    out: list[JournalRow] = []
    for row in rows:
        try:
            entry_ts = datetime.fromisoformat(row[0])
            exit_ts = datetime.fromisoformat(row[1])
        except (TypeError, ValueError):
            continue
        out.append(
            JournalRow(
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                symbol=row[2],
                qty=row[3],
                entry_price=row[4],
                exit_price=row[5],
                pnl_pct=row[6],
                pnl_usd=row[7],
                pnl_bucket=row[8],
                exit_reason=row[9],
                thesis=row[10],
            )
        )
    return out


def fetch_today_card(log_dir: Path) -> TodayCard:
    """Counts of today's decisions / orders / closed trips, plus realized P&L."""
    db_path = log_dir / "trades.sqlite"
    if not db_path.exists():
        return TodayCard(0, 0, 0, 0.0)
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_start.isoformat()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, isolation_level=None)
    except sqlite3.OperationalError:
        return TodayCard(0, 0, 0, 0.0)
    try:
        n_decisions = _scalar(
            conn,
            "SELECT COUNT(*) FROM trade_events "
            "WHERE event_type='floor_trader_decision' AND timestamp >= ?",
            (cutoff,),
        )
        n_orders = _scalar(
            conn,
            "SELECT COUNT(*) FROM trade_events "
            "WHERE event_type='order_submitted' AND timestamp >= ?",
            (cutoff,),
        )
        n_round_trips = _scalar(
            conn,
            "SELECT COUNT(*) FROM round_trips WHERE exit_ts >= ?",
            (cutoff,),
        )
        realized = _scalar(
            conn,
            "SELECT COALESCE(SUM(pnl_usd), 0.0) FROM round_trips WHERE exit_ts >= ?",
            (cutoff,),
            cast=float,
        )
    except sqlite3.OperationalError:
        return TodayCard(0, 0, 0, 0.0)
    finally:
        conn.close()
    return TodayCard(
        n_decisions=int(n_decisions),
        n_orders=int(n_orders),
        n_round_trips=int(n_round_trips),
        realized_pnl_today_usd=float(realized),
    )


def fetch_recent_decisions(log_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    """Last N floor-trader decisions for transparency on what the bot just thought."""
    db_path = log_dir / "trades.sqlite"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, isolation_level=None)
    except sqlite3.OperationalError:
        return []
    try:
        cur = conn.execute(
            """
            SELECT timestamp, symbol, payload
              FROM trade_events
             WHERE event_type='floor_trader_decision'
             ORDER BY timestamp DESC
             LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for ts, symbol, payload in rows:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        out.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "should_trade": data.get("should_trade"),
                "direction": data.get("direction"),
                "confidence": data.get("confidence"),
                "thesis": data.get("thesis"),
                "reason_for_pass": data.get("reason_for_pass"),
            }
        )
    return out


def kill_switch_active(cwd: Path | None = None) -> bool:
    """Check whether the file-based kill switch is currently set."""
    base = cwd or Path.cwd()
    return (base / "KILL_SWITCH").exists()


def set_kill_switch(active: bool, cwd: Path | None = None) -> bool:
    """Toggle the kill-switch file. Returns the new state."""
    base = cwd or Path.cwd()
    path = base / "KILL_SWITCH"
    if active:
        path.touch()
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
    return path.exists()


def _scalar(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...] = (),
    *,
    cast: type = int,
) -> Any:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    if row is None or row[0] is None:
        return cast(0)
    return cast(row[0])


def reconcile_now(log_dir: Path) -> int:
    """One-shot: rebuild round-trips from the journal. Returns count of new round-trips."""
    journal = TradeLogger(log_dir=log_dir)
    try:
        rec = RoundTripReconciler(journal)
        return len(rec.reconcile())
    finally:
        journal.close()


def aggregate_pnl(rows: list[JournalRow]) -> tuple[float, int, int]:
    """Sum realized P&L + count wins/losses across the supplied round-trips."""
    total = 0.0
    wins = losses = 0
    for r in rows:
        total += r.pnl_usd
        if r.pnl_bucket == PnlBucket.WIN.value:
            wins += 1
        elif r.pnl_bucket == PnlBucket.LOSS.value:
            losses += 1
    return total, wins, losses


# Public helper for templates.
def fmt_money(x: float | None) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}${x:,.2f}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 100:+.2f}%"


def fmt_time_ago(when: datetime) -> str:
    now = datetime.now(UTC) if when.tzinfo else datetime.now()
    delta = now - when
    if delta < timedelta(seconds=60):
        return f"{int(delta.total_seconds())}s ago"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)}m ago"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)}h ago"
    return f"{delta.days}d ago"
