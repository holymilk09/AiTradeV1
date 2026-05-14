"""EXP-005 — Consistency comparison: ungated vs VIX-gated bollinger.

Tests the central hypothesis from EXP-004: gating a mean-reversion entry
on volatility regime should improve cross-symbol Sharpe consistency
without giving up too much net expectancy.

For each (strategy, symbol):
  - Run walk-forward 180d/180d on 2024-2026 daily bars
  - Record: per-fold Sharpes, aggregate Sharpe mean/std, trade count,
    expectancy, train→test Sharpe correlation

Consistency metric: standard deviation of (test_sharpe_mean) ACROSS
SYMBOLS. Lower is better — means the strategy behaves similarly across
the panel.

Configurations tested:
  A. Base bollinger_reversion (period=20, num_std=1.5)  — control
  B. VIX-gated, rvol_pct_max=30  — strict regime gate
  C. VIX-gated, rvol_pct_max=50  — moderate gate (default)
  D. VIX-gated, rvol_pct_max=70  — permissive gate

Determinism check at the end: re-run A on AAPL → bit-identical metrics.
"""

from __future__ import annotations

import sys
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
from aitrade.strategy.examples.vix_gated_bollinger import VixGatedBollinger

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ", "TSLA", "META"]
START = "2024-01-01"
END = "2026-01-01"
TRAIN = pd.Timedelta(days=180)
TEST = pd.Timedelta(days=180)


def _factory(name: str, sym: str, *, threshold: float | None = None):
    if name == "base":
        return lambda: BollingerReversion(symbol=sym, period=20, num_std=1.5)
    if name == "gated":
        return lambda: VixGatedBollinger(
            symbol=sym, period=20, num_std=1.5,
            rvol_pct_max=threshold if threshold is not None else 50.0,
        )
    raise ValueError(name)


def run_panel(label: str, factory_fn) -> dict[str, dict[str, float]]:
    settings = get_settings()
    client = AlpacaDataClient(settings)
    cfg = BacktestConfig(timeframe=Timeframe.DAY_1)
    results: dict[str, dict[str, float]] = {}
    for sym in SYMBOLS:
        df = client.fetch_stock_bars(
            sym, Timeframe.DAY_1,
            datetime.fromisoformat(START).replace(tzinfo=UTC),
            datetime.fromisoformat(END).replace(tzinfo=UTC),
        )
        if df.empty:
            continue
        bars = list(df_to_bars(df, sym))
        try:
            s = walk_forward(factory_fn(sym), bars, train=TRAIN, test=TEST, config=cfg)
        except ValueError:
            continue
        results[sym] = {
            "sharpe_mean": round(s.test_sharpe_mean, 3),
            "sharpe_std": round(s.test_sharpe_std, 3),
            "trades": s.test_trade_count_total,
            "expect": round(s.test_expectancy_mean, 1),
            "corr": round(s.train_test_sharpe_corr, 3),
        }
    return results


def consistency_metrics(results: dict[str, dict[str, float]]) -> dict[str, float]:
    if not results:
        return {"panel_mean_sharpe": 0.0, "panel_std_sharpe": 0.0, "n": 0}
    sharpes = [r["sharpe_mean"] for r in results.values()]
    expects = [r["expect"] for r in results.values()]
    trades = [r["trades"] for r in results.values()]
    return {
        "panel_mean_sharpe": round(fmean(sharpes), 3),
        "panel_std_sharpe": round(pstdev(sharpes) if len(sharpes) > 1 else 0.0, 3),
        "panel_min_sharpe": round(min(sharpes), 3),
        "panel_max_sharpe": round(max(sharpes), 3),
        "n_with_pos_sharpe": sum(1 for s in sharpes if s > 0.5),
        "n_with_mean_gt_std": sum(
            1 for r in results.values() if r["sharpe_mean"] > r["sharpe_std"]
        ),
        "total_trades": sum(trades),
        "mean_expectancy": round(fmean(expects), 1),
        "n": len(results),
    }


configs = [
    ("BASE bollinger (period=20, std=1.5)",
     lambda sym: _factory("base", sym)),
    ("GATED rvol_pct_max=30",
     lambda sym: _factory("gated", sym, threshold=30.0)),
    ("GATED rvol_pct_max=50",
     lambda sym: _factory("gated", sym, threshold=50.0)),
    ("GATED rvol_pct_max=70",
     lambda sym: _factory("gated", sym, threshold=70.0)),
]

print(f"=== EXP-005 consistency test · {len(SYMBOLS)} symbols · {START} → {END} ===\n")
summary: list[tuple[str, dict[str, float]]] = []
for label, fn in configs:
    print(f"--- {label} ---")
    per_sym = run_panel(label, fn)
    for sym in SYMBOLS:
        r = per_sym.get(sym)
        if r:
            print(f"  {sym:<6} sharpe={r['sharpe_mean']:>+.2f} (std {r['sharpe_std']:>.2f}) "
                  f"trades={r['trades']:>3} expect={r['expect']:>+7.1f} corr={r['corr']:>+.2f}")
        else:
            print(f"  {sym:<6} (no data)")
    m = consistency_metrics(per_sym)
    summary.append((label, m))
    print(f"  CONSISTENCY: mean={m['panel_mean_sharpe']:>+.2f}  "
          f"std={m['panel_std_sharpe']:>.2f}  "
          f"min={m['panel_min_sharpe']:>+.2f}  "
          f"max={m['panel_max_sharpe']:>+.2f}  "
          f"pass_rubric={m['n_with_mean_gt_std']}/{m['n']}  "
          f"trades={m['total_trades']}  "
          f"expect={m['mean_expectancy']:>+.1f}\n")

# Rank configs by consistency (lower std is better) then by mean Sharpe.
print("\n=== SUMMARY ===")
print(f"{'config':<42} {'mean':>7} {'std':>7} {'pass':>6} {'trades':>7} {'expect':>9}")
_sort_key = lambda x: (x[1]['panel_std_sharpe'], -x[1]['panel_mean_sharpe'])  # noqa: E731
for label, m in sorted(summary, key=_sort_key):
    print(f"{label:<42} {m['panel_mean_sharpe']:>+7.2f} {m['panel_std_sharpe']:>7.2f} "
          f"{m['n_with_mean_gt_std']:>2d}/{m['n']:<2d}  "
          f"{m['total_trades']:>7d} {m['mean_expectancy']:>+9.1f}")

# Determinism check — same backtest twice should be bit-identical.
print("\n=== DETERMINISM CHECK ===")
settings = get_settings()
client = AlpacaDataClient(settings)
df = client.fetch_stock_bars(
    "AAPL", Timeframe.DAY_1,
    datetime.fromisoformat(START).replace(tzinfo=UTC),
    datetime.fromisoformat(END).replace(tzinfo=UTC),
)
bars = list(df_to_bars(df, "AAPL"))
cfg = BacktestConfig(timeframe=Timeframe.DAY_1)
runs = []
for _i in range(3):
    s = walk_forward(
        lambda: BollingerReversion(symbol="AAPL", period=20, num_std=1.5),
        bars, train=TRAIN, test=TEST, config=cfg,
    )
    runs.append((round(s.test_sharpe_mean, 6), round(s.test_expectancy_mean, 6),
                 s.test_trade_count_total))
print(f"3 runs of base bollinger on AAPL: {runs}")
if len(set(runs)) == 1:
    print("  ✓ DETERMINISTIC — all 3 runs bit-identical")
else:
    print("  ✗ NON-DETERMINISTIC — runs differ, investigate")
    sys.exit(2)
