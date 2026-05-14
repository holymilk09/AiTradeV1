"""Param sweep over BollingerReversion (period, num_std).

One symbol, one date range. Fetches historical bars ONCE then re-uses
them across every grid cell — the walk_forward call is cheap once the
bars are in memory.

Usage:
    ./scripts/atr -- dummy  # just to remind you to use uv via the wrapper
    ./.venv/bin/python scripts/sweep_bollinger.py [SYMBOL]

Output:
    Grid table on stdout AND data/research/sweep_bollinger_<SYMBOL>.csv
"""

from __future__ import annotations

import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.backtest.simple_runner import BacktestConfig
from aitrade.backtest.walk_forward import walk_forward
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
START = "2024-01-01"
END = "2026-01-01"
TRAIN = pd.Timedelta(days=180)
TEST = pd.Timedelta(days=180)
PERIODS = [10, 14, 20, 30, 50]
NUM_STDS = [1.5, 2.0, 2.5, 3.0]

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
bars = list(df_to_bars(df, SYMBOL))
cfg = BacktestConfig(timeframe=Timeframe.DAY_1)

print(f"Sweep BollingerReversion on {SYMBOL} [{START} → {END}] · train=180d · test=180d")
print(f"{len(bars)} bars loaded")
print()
header = ("period", "num_std", "sharpe_mean", "sharpe_std", "mean>std",
          "trades", "expect_usd", "tt_corr", "ddworst%")
print("{:>7} {:>8} {:>12} {:>11} {:>9} {:>7} {:>11} {:>8} {:>9}".format(*header))
print("-" * 92)

out_dir = Path("data/research")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / f"sweep_bollinger_{SYMBOL}.csv"

with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    for p in PERIODS:
        for k in NUM_STDS:
            try:
                summary = walk_forward(
                    lambda p=p, k=k: BollingerReversion(
                        symbol=SYMBOL, period=p, num_std=k
                    ),
                    bars,
                    train=TRAIN,
                    test=TEST,
                    config=cfg,
                )
                row = [
                    p, k,
                    round(summary.test_sharpe_mean, 2),
                    round(summary.test_sharpe_std, 2),
                    "yes" if summary.test_sharpe_mean > summary.test_sharpe_std else "no",
                    summary.test_trade_count_total,
                    round(summary.test_expectancy_mean, 2),
                    round(summary.train_test_sharpe_corr, 2),
                    round(summary.test_max_dd_pct_worst, 2),
                ]
                print("{:>7} {:>8} {:>12} {:>11} {:>9} {:>7} {:>11} {:>8} {:>9}".format(*row))
                w.writerow(row)
            except Exception as e:
                err_row = [p, k, "ERR", "", "", "", str(e)[:40], "", ""]
                print("{:>7} {:>8} {:>12} {:>11} {:>9} {:>7} {:>11} {:>8} {:>9}".format(*err_row))
                w.writerow(err_row)

print()
print(f"wrote {out_csv}")
