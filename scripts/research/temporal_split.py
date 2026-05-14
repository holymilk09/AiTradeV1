"""EXP-006 — Temporal stability split.

Goal: test whether bollinger_reversion's cross-symbol consistency
survives a regime split. Same 8-symbol panel as EXP-001/005, but
walk-forward on TWO non-overlapping date ranges:

  H1 = 2024-01-01 → 2025-01-01  (one year)
  H2 = 2025-01-01 → 2026-01-01  (one year)

For each half, use 120d train / 120d test windows. That's ~2 folds per
half per symbol — small, but the question is whether the *direction*
of the edge is stable, not whether each fold is statistically clean.

Output:
  - Per-symbol Sharpe in H1 and H2
  - Per-symbol Sharpe delta (H2 - H1) — large flips = regime-fragile
  - Panel mean Sharpe in H1 vs H2 — large gap = global regime shift
  - Symbols where both halves pass the rubric (most robust)
"""

from __future__ import annotations

from datetime import UTC, datetime
from statistics import fmean, pstdev

import pandas as pd

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.backtest.simple_runner import BacktestConfig
from aitrade.backtest.walk_forward import walk_forward
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ", "TSLA", "META"]
HALVES = [
    ("H1 2024", "2024-01-01", "2025-01-01"),
    ("H2 2025", "2025-01-01", "2026-01-01"),
]
TRAIN = pd.Timedelta(days=120)
TEST = pd.Timedelta(days=120)


def run_one(sym: str, start: str, end: str) -> dict[str, float] | None:
    settings = get_settings()
    client = AlpacaDataClient(settings)
    df = client.fetch_stock_bars(
        sym, Timeframe.DAY_1,
        datetime.fromisoformat(start).replace(tzinfo=UTC),
        datetime.fromisoformat(end).replace(tzinfo=UTC),
    )
    if df.empty:
        return None
    bars = list(df_to_bars(df, sym))
    try:
        s = walk_forward(
            lambda: BollingerReversion(symbol=sym, period=20, num_std=1.5),
            bars,
            train=TRAIN, test=TEST,
            config=BacktestConfig(timeframe=Timeframe.DAY_1),
        )
    except ValueError:
        return None
    return {
        "sharpe": round(s.test_sharpe_mean, 3),
        "std": round(s.test_sharpe_std, 3),
        "trades": s.test_trade_count_total,
        "expect": round(s.test_expectancy_mean, 1),
        "folds": len(s.folds),
    }


# Collect.
all_results: dict[str, dict[str, dict[str, float] | None]] = {sym: {} for sym in SYMBOLS}
for sym in SYMBOLS:
    for label, start, end in HALVES:
        all_results[sym][label] = run_one(sym, start, end)

# Print per-symbol comparison.
print(f"=== EXP-006 · temporal split · 120d/120d walk-forward ===\n")
print(f"{'SYM':<6} {'H1 Sharpe':>11} {'H1 trades':>10} {'H1 expect':>10}  "
      f"{'H2 Sharpe':>11} {'H2 trades':>10} {'H2 expect':>10}  "
      f"{'delta':>7} {'sign_flip?':>10}")
flips = 0
both_pos = 0
both_pass_rubric = 0
for sym in SYMBOLS:
    h1 = all_results[sym].get("H1 2024")
    h2 = all_results[sym].get("H2 2025")
    if not h1 or not h2:
        print(f"{sym:<6}  (missing data)")
        continue
    s1, s2 = h1["sharpe"], h2["sharpe"]
    delta = s2 - s1
    flipped = (s1 > 0) != (s2 > 0)
    if flipped:
        flips += 1
    if s1 > 0 and s2 > 0:
        both_pos += 1
    if h1["sharpe"] > h1["std"] and h2["sharpe"] > h2["std"]:
        both_pass_rubric += 1
    print(f"{sym:<6} {s1:>+11.2f} {h1['trades']:>10d} {h1['expect']:>+10.1f}  "
          f"{s2:>+11.2f} {h2['trades']:>10d} {h2['expect']:>+10.1f}  "
          f"{delta:>+7.2f} {'YES' if flipped else 'no':>10}")

# Panel-level metrics per half.
print()
print("=== panel-level per half ===")
for label, _, _ in HALVES:
    sharpes = [
        all_results[sym][label]["sharpe"]
        for sym in SYMBOLS
        if all_results[sym][label] is not None
    ]
    trades = [
        all_results[sym][label]["trades"]
        for sym in SYMBOLS
        if all_results[sym][label] is not None
    ]
    if sharpes:
        print(f"{label}: mean={fmean(sharpes):+.2f}  "
              f"std={pstdev(sharpes) if len(sharpes) > 1 else 0:.2f}  "
              f"trades_total={sum(trades)}  "
              f"n_with_pos={sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

print()
print(f"sign flips between halves: {flips}/{len(SYMBOLS)}")
print(f"both halves positive: {both_pos}/{len(SYMBOLS)}")
print(f"both halves pass rubric (mean > std): {both_pass_rubric}/{len(SYMBOLS)}")
