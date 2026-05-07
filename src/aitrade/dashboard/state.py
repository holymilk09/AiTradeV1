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


# --- Phase 7 dashboard rebuild — additional data sources --------------------


@dataclass(frozen=True, slots=True)
class StrategyStatRow:
    strategy_id: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    total_pnl_usd: float
    avg_pnl_usd: float


@dataclass(frozen=True, slots=True)
class EventRow:
    event_id: str
    event_type: str
    symbol: str
    timestamp: datetime
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateRow:
    """Trimmed candidate row for the Live tab — pulled from the most-recent
    CANDIDATE_BOARD event in the journal."""

    symbol: str
    combined_score: float
    buzz_score: float
    pattern_score: float
    trend_score: float
    pattern_hits: list[str]


def _ro_conn(log_dir: Path) -> sqlite3.Connection | None:
    """One-shot read-only sqlite handle, or None if the journal isn't there yet."""
    db_path = log_dir / "trades.sqlite"
    if not db_path.exists():
        return None
    try:
        return sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, isolation_level=None
        )
    except sqlite3.OperationalError:
        return None


def fetch_strategy_stats(log_dir: Path) -> list[StrategyStatRow]:
    """Per-strategy roll-up over closed round-trips."""
    conn = _ro_conn(log_dir)
    if conn is None:
        return []
    try:
        cur = conn.execute(
            "SELECT strategy_id, pnl_usd, pnl_bucket FROM round_trips"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    by_strat: dict[str, dict[str, float]] = {}
    for sid, pnl_usd, bucket in rows:
        s = sid or "unknown"
        slot = by_strat.setdefault(
            s, {"n": 0, "wins": 0, "losses": 0, "total": 0.0}
        )
        slot["n"] += 1
        with contextlib.suppress(TypeError, ValueError):
            slot["total"] += float(pnl_usd)
        if str(bucket) == PnlBucket.WIN.value:
            slot["wins"] += 1
        elif str(bucket) == PnlBucket.LOSS.value:
            slot["losses"] += 1

    out: list[StrategyStatRow] = []
    for strat, slot in by_strat.items():
        n = int(slot["n"])
        wins = int(slot["wins"])
        losses = int(slot["losses"])
        total = float(slot["total"])
        out.append(
            StrategyStatRow(
                strategy_id=strat,
                n_trades=n,
                n_wins=wins,
                n_losses=losses,
                win_rate=wins / n if n else 0.0,
                total_pnl_usd=total,
                avg_pnl_usd=total / n if n else 0.0,
            )
        )
    out.sort(key=lambda r: r.n_trades, reverse=True)
    return out


def fetch_recent_events(
    log_dir: Path,
    *,
    event_type: str | None = None,
    limit: int = 100,
) -> list[EventRow]:
    """Recent events from the journal SQLite mirror, newest first."""
    conn = _ro_conn(log_dir)
    if conn is None:
        return []
    try:
        if event_type:
            cur = conn.execute(
                "SELECT event_id, event_type, symbol, timestamp, payload "
                "FROM trade_events WHERE event_type = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (event_type, limit),
            )
        else:
            cur = conn.execute(
                "SELECT event_id, event_type, symbol, timestamp, payload "
                "FROM trade_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    out: list[EventRow] = []
    for event_id, etype, symbol, ts, payload in rows:
        try:
            ts_dt = datetime.fromisoformat(ts)
        except (TypeError, ValueError):
            continue
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {"raw": payload}
        out.append(
            EventRow(
                event_id=event_id,
                event_type=etype,
                symbol=symbol or "-",
                timestamp=ts_dt,
                payload=data if isinstance(data, dict) else {"value": data},
            )
        )
    return out


def fetch_event_types(log_dir: Path) -> list[str]:
    """Distinct event_type values present in the journal — for filter UI."""
    conn = _ro_conn(log_dir)
    if conn is None:
        return []
    try:
        cur = conn.execute(
            "SELECT DISTINCT event_type FROM trade_events ORDER BY event_type"
        )
        return [str(row[0]) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def fetch_latest_candidate_board(
    log_dir: Path,
    *,
    limit: int = 20,
) -> tuple[datetime | None, list[CandidateRow]]:
    """Most-recent CANDIDATE_BOARD event projected into ``CandidateRow``s."""
    events = fetch_recent_events(log_dir, event_type="candidate_board", limit=1)
    if not events:
        return None, []
    payload = events[0].payload
    raw = payload.get("candidates", [])
    if not isinstance(raw, list):
        return events[0].timestamp, []
    rows: list[CandidateRow] = []
    for c in raw[:limit]:
        if not isinstance(c, dict):
            continue
        rows.append(
            CandidateRow(
                symbol=str(c.get("symbol", "?")),
                combined_score=float(c.get("combined_score", 0.0) or 0.0),
                buzz_score=float(c.get("buzz_score", 0.0) or 0.0),
                pattern_score=float(c.get("pattern_score", 0.0) or 0.0),
                trend_score=float(c.get("trend_score", 0.0) or 0.0),
                pattern_hits=[str(h) for h in c.get("pattern_hits", []) or []],
            )
        )
    return events[0].timestamp, rows


def fetch_latest_market_snapshot(log_dir: Path) -> dict[str, Any] | None:
    """Most-recent MARKET_SNAPSHOT event payload, or None."""
    events = fetch_recent_events(log_dir, event_type="market_snapshot", limit=1)
    if not events:
        return None
    return events[0].payload


def fetch_filtered_round_trips(
    log_dir: Path,
    *,
    symbol: str | None = None,
    pattern: str | None = None,
    bucket: str | None = None,
    regime: str | None = None,
    limit: int = 200,
) -> list[JournalRow]:
    """History tab: round-trips with optional filters, newest first."""
    conn = _ro_conn(log_dir)
    if conn is None:
        return []
    try:
        clauses: list[str] = []
        params: list[object] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if pattern:
            clauses.append("pattern_hits LIKE ?")
            params.append(f'%"{pattern}"%')
        if bucket:
            clauses.append("pnl_bucket = ?")
            params.append(bucket.upper())
        if regime:
            clauses.append("market_snapshot LIKE ?")
            params.append(f'%"regime": "{regime}"%')
        sql = (
            "SELECT entry_ts, exit_ts, symbol, qty, entry_price, exit_price, "
            "pnl_pct, pnl_usd, pnl_bucket, exit_reason, entry_thesis "
            "FROM round_trips"
        )
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY exit_ts DESC LIMIT ?"
        params.append(int(limit))
        cur = conn.execute(sql, params)
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
