"""E1 — Overnight vs intraday return decomposition.

Hypothesis: Buying SPY at close and selling at next open captures
nearly all of SPY's long-run drift with materially lower variance
than buy-and-hold. The intraday session (open → close) has been
roughly zero-mean since ~2010 in published academic studies
(Lou-Polk-Skouras 2019, Cliff-Cooper-Gulen 2008).

If it replicates on 2020-2026 SPY daily bars, this is a real
microstructure edge that can be wired into the engine — and it has
the rare property that even a basic implementation beats the
market on a Sharpe basis.

Cost model: 1bp fee + 2bp slippage per leg, so 6bps round-trip per
day. At 252 days/yr that's 15.1% drag — a real test of whether the
overnight edge survives transaction cost.

Usage:
    ./.venv/bin/python scripts/research/overnight_vs_intraday.py [SYMBOL]

Default symbol: SPY. Try QQQ, IWM, individual large-caps for comparison.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from math import sqrt

import pandas as pd

from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "SPY"
START = "2020-01-01"
END = "2026-01-01"
ROUND_TRIP_BPS = 6.0  # 1bp fee + 2bp slippage per leg, 2 legs per day

settings = get_settings()
client = AlpacaDataClient(settings)
df = client.fetch_stock_bars(
    SYMBOL,
    Timeframe.DAY_1,
    datetime.fromisoformat(START).replace(tzinfo=UTC),
    datetime.fromisoformat(END).replace(tzinfo=UTC),
)
if df.empty:
    print(f"no bars for {SYMBOL}")
    sys.exit(1)

# Normalize to one row per session. Alpaca returns 'open', 'high', 'low',
# 'close', 'volume' on bars. df index is timestamp.
df = df.sort_index().reset_index()
df["prev_close"] = df["close"].shift(1)
df["overnight_ret"] = df["open"] / df["prev_close"] - 1.0
df["intraday_ret"] = df["close"] / df["open"] - 1.0
df["full_ret"] = df["close"] / df["prev_close"] - 1.0
df = df.dropna().reset_index(drop=True)

# Cost-adjusted overnight strategy: pay 6bps each round-trip day.
df["overnight_ret_net"] = df["overnight_ret"] - ROUND_TRIP_BPS / 10_000.0


def stats(returns: pd.Series, label: str) -> dict[str, float]:
    n = len(returns)
    if n < 2:
        return {"label": label, "n": n}
    mean = float(returns.mean())
    sd = float(returns.std(ddof=1))
    ann_ret = mean * 252.0
    ann_sd = sd * sqrt(252.0)
    sharpe = ann_ret / ann_sd if ann_sd > 0 else 0.0
    cum = float((1 + returns).prod() - 1.0)
    pos = float((returns > 0).mean())
    worst = float(returns.min())
    best = float(returns.max())
    return {
        "label": label,
        "n": n,
        "mean_bps_per_day": mean * 10_000.0,
        "ann_ret_pct": ann_ret * 100.0,
        "ann_vol_pct": ann_sd * 100.0,
        "sharpe": sharpe,
        "cum_ret_pct": cum * 100.0,
        "hit_pct": pos * 100.0,
        "worst_bp": worst * 10_000.0,
        "best_bp": best * 10_000.0,
    }


rows = [
    stats(df["overnight_ret"], "overnight (close→open, gross)"),
    stats(df["overnight_ret_net"], "overnight (close→open, net of 6bps)"),
    stats(df["intraday_ret"], "intraday (open→close, no cost — buy-and-hold proxy)"),
    stats(df["full_ret"], "full session (prev_close→close, buy-and-hold)"),
]

print(f"=== {SYMBOL} return decomposition · {START} → {END} ===")
print(f"sessions: {len(df)}\n")
hdr = (f"{'series':<46} {'n':>5} {'μbps':>7} {'ann%':>7} {'σann%':>7} "
       f"{'Sharpe':>7} {'cum%':>9} {'hit%':>6}")
print(hdr)
for r in rows:
    print(
        f"{r['label']:<46} {r['n']:>5d} "
        f"{r['mean_bps_per_day']:>7.2f} "
        f"{r['ann_ret_pct']:>7.2f} "
        f"{r['ann_vol_pct']:>7.2f} "
        f"{r['sharpe']:>7.2f} "
        f"{r['cum_ret_pct']:>9.2f} "
        f"{r['hit_pct']:>6.1f}"
    )

# Worst/best day comparison — a real floor trader cares about tail.
print()
print("Tails:")
for col, lbl in [
    ("overnight_ret", "overnight"),
    ("intraday_ret", "intraday"),
    ("full_ret", "full"),
]:
    s = df[col]
    p1 = s.quantile(0.01) * 10_000
    p99 = s.quantile(0.99) * 10_000
    print(f"  {lbl:<10} p1={p1:>8.1f}bps  p99={p99:>8.1f}bps  worst={s.min()*10_000:>8.1f}bps")

# Correlation matrix between segments — are overnight and intraday returns
# negatively correlated? (If so, they'd be diversifying when combined.)
corr = df[["overnight_ret", "intraday_ret"]].corr().iloc[0, 1]
print()
print(f"corr(overnight, intraday) = {corr:+.3f}")
print(
    "→ negative means overnight & intraday mean-revert against each other "
    "(overnight gain often gives back intraday — classic overnight drift signature)."
)
