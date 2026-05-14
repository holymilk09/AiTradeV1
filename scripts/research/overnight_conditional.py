"""E1b — Conditional overnight strategy.

Unconditional overnight on SPY/QQQ has gross edge but doesn't survive
round-trip cost. Question: can we *filter* the nights and only trade
when the expected overnight return materially exceeds cost?

Filters tested:
  - prev day's intraday return sign  (was today red? if so, overnight bounce)
  - prev day's intraday return magnitude (large red → bigger bounce?)
  - VIXY level (regime proxy — high vol = bigger overnight risk premium)
  - day-of-week (Mon-Fri buckets)

For each filter: split the dataset, compute conditional overnight Sharpe
+ net-of-cost mean, AND the number of nights selected (you need enough
to make it tradable, not p-hack).

Usage:
    ./.venv/bin/python scripts/research/overnight_conditional.py [SYMBOL]
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from math import sqrt

import pandas as pd

from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "QQQ"
START = "2020-01-01"
END = "2026-01-01"
ROUND_TRIP_BPS = 6.0

settings = get_settings()
client = AlpacaDataClient(settings)


def fetch(sym: str) -> pd.DataFrame:
    df = client.fetch_stock_bars(
        sym,
        Timeframe.DAY_1,
        datetime.fromisoformat(START).replace(tzinfo=UTC),
        datetime.fromisoformat(END).replace(tzinfo=UTC),
    )
    if df.empty:
        return df
    df = df.sort_index().reset_index()
    df["prev_close"] = df["close"].shift(1)
    df["overnight_ret"] = df["open"] / df["prev_close"] - 1.0
    df["intraday_ret"] = df["close"] / df["open"] - 1.0
    df["prev_intraday"] = df["intraday_ret"].shift(1)
    return df.dropna().reset_index(drop=True)


def summarize(returns: pd.Series, label: str, n_total: int) -> str:
    n = len(returns)
    if n < 5:
        return f"{label:<40} n={n:>5d}  (too few samples)"
    mean = float(returns.mean())
    sd = float(returns.std(ddof=1))
    sharpe = (mean * 252.0) / (sd * sqrt(252.0)) if sd > 0 else 0.0
    net_mean = mean - ROUND_TRIP_BPS / 10_000.0
    net_ann = net_mean * 252.0
    pct_used = (n / n_total) * 100.0
    return (
        f"{label:<40} n={n:>5d} ({pct_used:>5.1f}%)  "
        f"μgross={mean*10_000:>+6.2f}bps  "
        f"Sharpe={sharpe:>+5.2f}  "
        f"netμ={net_mean*10_000:>+6.2f}bps  net_ann={net_ann*100:>+6.2f}%"
    )


df = fetch(SYMBOL)
if df.empty:
    print(f"no bars for {SYMBOL}")
    sys.exit(1)

# Pull VIXY for regime conditioning. Align on date.
vixy = fetch("VIXY")
if not vixy.empty:
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    vixy["date"] = pd.to_datetime(vixy["timestamp"]).dt.date
    df = df.merge(vixy[["date", "close"]].rename(columns={"close": "vixy"}), on="date", how="left")
    df["vixy_z"] = (df["vixy"] - df["vixy"].rolling(60).mean()) / df["vixy"].rolling(60).std()
    df["vixy_pct"] = df["vixy"].rolling(252, min_periods=60).apply(
        lambda s: (s.rank().iloc[-1] / len(s)) * 100.0
    )
df = df.dropna(subset=["overnight_ret", "prev_intraday"]).reset_index(drop=True)
n_total = len(df)

print(f"=== {SYMBOL} conditional overnight · {START} → {END} ===")
print(f"sessions with prior-day intraday: {n_total}")
print(f"round-trip cost assumed: {ROUND_TRIP_BPS:.1f} bps\n")

# Baseline.
print(summarize(df["overnight_ret"], "ALL nights (baseline)", n_total))
print()

# Filter 1 — prior day intraday return SIGN.
print("Filter 1: prior intraday direction")
print(summarize(df[df["prev_intraday"] < 0]["overnight_ret"], "  prev intraday RED", n_total))
print(summarize(df[df["prev_intraday"] >= 0]["overnight_ret"], "  prev intraday GREEN", n_total))
print()

# Filter 2 — prior intraday magnitude (z-score).
print("Filter 2: prior intraday MAGNITUDE (red days only)")
red = df[df["prev_intraday"] < 0]
red_std = red["prev_intraday"].std()
print(summarize(red[red["prev_intraday"] < -red_std * 1.0]["overnight_ret"],
                "  prev intraday < -1σ red", n_total))
print(summarize(red[red["prev_intraday"] < -red_std * 1.5]["overnight_ret"],
                "  prev intraday < -1.5σ red", n_total))
print(summarize(red[red["prev_intraday"] < -red_std * 2.0]["overnight_ret"],
                "  prev intraday < -2σ red", n_total))
print()

# Filter 3 — VIXY level (regime).
if "vixy_pct" in df.columns:
    print("Filter 3: VIXY rolling percentile (last 252d)")
    vd = df.dropna(subset=["vixy_pct"])
    print(summarize(
        vd[vd["vixy_pct"] >= 80]["overnight_ret"], "  VIXY top quintile (>=80%)", n_total))
    print(summarize(
        vd[vd["vixy_pct"] >= 90]["overnight_ret"], "  VIXY top decile (>=90%)", n_total))
    print(summarize(
        vd[vd["vixy_pct"] <= 20]["overnight_ret"], "  VIXY bottom quintile (<=20%)", n_total))
    print()

# Filter 4 — day of week.
df["dow"] = pd.to_datetime(df["timestamp"]).dt.day_name()
print("Filter 4: day of week (overnight HELD INTO this day)")
for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
    sub = df[df["dow"] == day]
    print(summarize(sub["overnight_ret"], f"  {day}", n_total))
print()

# Composite: prev intraday red + VIXY top quintile.
if "vixy_pct" in df.columns:
    compo = df[(df["prev_intraday"] < 0) & (df["vixy_pct"] >= 70)]
    print("Composite filter: prev RED + VIXY ≥70th pct")
    print(summarize(compo["overnight_ret"], "  composite filter", n_total))
