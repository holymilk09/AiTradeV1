"""Weekly review: paper session analytics from the trade journal.

Usage:
    uv run python scripts/weekly_review.py --since 2025-01-01

Reads the JSONL trade log written by ``aitrade.logging.trade_logger``
and reports per-strategy:

- round-trip count + win rate + expectancy
- equity curve (cumulative realized P&L)
- expectancy week-over-week (drift signal)

No plotting deps — output is text + a parquet of round-trips for any
notebook to consume. Designed to run unattended via launchd or cron.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True, slots=True)
class RoundTrip:
    strategy: str
    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    qty: float
    entry_px: float
    exit_px: float
    pnl: float


def load_events(jsonl_path: Path, since: datetime | None = None) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for raw in f:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ts = pd.to_datetime(ev.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            if since and ts < pd.Timestamp(since):
                continue
            ev["timestamp"] = ts
            rows.append(ev)
    return pd.DataFrame(rows)


def round_trips(events: pd.DataFrame) -> list[RoundTrip]:
    """FIFO-match fills per (strategy, symbol) into long-only round trips."""
    if events.empty:
        return []
    fills = events[events["event_type"] == "fill_received"]
    open_legs: dict[tuple[str, str], deque[tuple[datetime, float, float]]] = defaultdict(deque)
    trips: list[RoundTrip] = []
    for _, row in fills.iterrows():
        payload = row["payload"]
        if not isinstance(payload, dict):
            continue
        fill = payload.get("fill") or payload
        side = (fill.get("side") or "").lower()
        qty = float(fill.get("filled_qty") or fill.get("qty") or 0)
        px = float(fill.get("avg_fill_price") or fill.get("price") or 0)
        if qty <= 0 or px <= 0:
            continue
        key = (row.get("strategy_id") or "", row.get("symbol") or "")
        ts = row["timestamp"].to_pydatetime()
        if side == "buy":
            open_legs[key].append((ts, qty, px))
        elif side == "sell":
            remaining = qty
            while remaining > 0 and open_legs[key]:
                entry_ts, lot_qty, lot_px = open_legs[key][0]
                close_qty = min(remaining, lot_qty)
                trips.append(
                    RoundTrip(
                        strategy=key[0],
                        symbol=key[1],
                        entry_ts=entry_ts,
                        exit_ts=ts,
                        qty=close_qty,
                        entry_px=lot_px,
                        exit_px=px,
                        pnl=(px - lot_px) * close_qty,
                    )
                )
                remaining -= close_qty
                if close_qty == lot_qty:
                    open_legs[key].popleft()
                else:
                    open_legs[key][0] = (entry_ts, lot_qty - close_qty, lot_px)
    return trips


def summarize(trips: list[RoundTrip]) -> pd.DataFrame:
    if not trips:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "strategy": t.strategy,
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "qty": t.qty,
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "pnl": t.pnl,
                "hold_secs": (t.exit_ts - t.entry_ts).total_seconds(),
            }
            for t in trips
        ]
    )
    # Drop tz before period conversion (pandas Period is tz-naive).
    df["week"] = df["exit_ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W").astype(
        str
    )
    return df


def per_strategy_stats(rt: pd.DataFrame) -> pd.DataFrame:
    if rt.empty:
        return pd.DataFrame()
    g = rt.groupby("strategy")
    return pd.DataFrame(
        {
            "trades": g["pnl"].count(),
            "win_rate": g["pnl"].apply(lambda s: float((s > 0).mean())),
            "expectancy": g["pnl"].mean(),
            "total_pnl": g["pnl"].sum(),
            "median_pnl": g["pnl"].median(),
            "worst": g["pnl"].min(),
            "best": g["pnl"].max(),
        }
    ).round(2)


def weekly_expectancy(rt: pd.DataFrame) -> pd.DataFrame:
    if rt.empty:
        return pd.DataFrame()
    return (
        rt.groupby(["strategy", "week"])["pnl"]
        .agg(["count", "mean", "sum"])
        .rename(columns={"count": "trades", "mean": "expectancy", "sum": "pnl"})
        .reset_index()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, default=Path("logs/trades.jsonl"))
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--out", type=Path, default=Path("runs/weekly_review"))
    args = p.parse_args()

    since = (
        datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    )
    events = load_events(args.logs, since=since)
    trips = round_trips(events)
    rt = summarize(trips)
    stats = per_strategy_stats(rt)
    weekly = weekly_expectancy(rt)

    console = Console()
    console.print(f"Loaded {len(events)} events; {len(trips)} round-trips.")
    if stats.empty:
        console.print("[yellow]No closed round-trips yet.[/yellow]")
        return

    t = Table(title="Per-strategy stats")
    for col in stats.columns:
        t.add_column(col, justify="right")
    t.add_column("strategy")
    for strategy, row in stats.iterrows():
        t.add_row(*[f"{v}" for v in row.values], strategy)
    console.print(t)

    args.out.mkdir(parents=True, exist_ok=True)
    rt.to_parquet(args.out / "round_trips.parquet")
    weekly.to_parquet(args.out / "weekly.parquet")
    console.print(f"[green]Wrote {args.out / 'round_trips.parquet'}[/green]")


if __name__ == "__main__":
    main()
