"""EXP-007 — Strategy leaderboard.

Head-to-head every registered strategy against the same 8-symbol panel,
same date range, same train/test windows. Produces ONE ranked table
ordered by a composite consistency score:

   rank_score = panel_mean_sharpe / max(panel_std_sharpe, 0.1)

A strategy with high mean Sharpe but high cross-symbol std is fragile;
a strategy with moderate mean but tight std is consistent. The rank
combines both — call it "consistency-adjusted return."

Also reports:
  - per-symbol Sharpe (so you can see WHERE each strategy works)
  - panel pass-rate (n symbols passing the mean>std rubric)
  - total trades (low = strategy barely fires; high = signal-rich)
  - panel mean expectancy ($ per trade)

This is the kind of comparison that should drive strategy choices.
Everything else (param sweeps, regime gating) is downstream of "is this
strategy worth keeping at all on this universe."
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
from aitrade.strategy.registry import get_strategy, list_strategies

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ", "TSLA", "META"]
START = "2024-01-01"
END = "2026-01-01"
TRAIN = pd.Timedelta(days=180)
TEST = pd.Timedelta(days=180)


def run_for(strategy_name: str, sym: str, bars) -> dict[str, float] | None:
    try:
        s = walk_forward(
            lambda: get_strategy(strategy_name, symbol=sym),
            bars,
            train=TRAIN, test=TEST,
            config=BacktestConfig(timeframe=Timeframe.DAY_1),
        )
    except (ValueError, Exception):
        return None
    return {
        "sharpe": round(s.test_sharpe_mean, 2),
        "std": round(s.test_sharpe_std, 2),
        "trades": s.test_trade_count_total,
        "expect": round(s.test_expectancy_mean, 1),
        "pass": s.test_sharpe_mean > s.test_sharpe_std,
        "corr": round(s.train_test_sharpe_corr, 2),
    }


def main() -> None:
    settings = get_settings()
    client = AlpacaDataClient(settings)

    # Fetch bars ONCE per symbol.
    bars_by_sym: dict[str, list] = {}
    for sym in SYMBOLS:
        df = client.fetch_stock_bars(
            sym, Timeframe.DAY_1,
            datetime.fromisoformat(START).replace(tzinfo=UTC),
            datetime.fromisoformat(END).replace(tzinfo=UTC),
        )
        if not df.empty:
            bars_by_sym[sym] = list(df_to_bars(df, sym))

    strategies = list_strategies()
    print("=== EXP-007 strategy leaderboard ===")
    print(f"universe: {', '.join(SYMBOLS)}")
    print(f"range: {START} → {END}  ·  train=180d/test=180d daily")
    print(f"strategies: {', '.join(strategies)}\n")

    # Per-strategy results table.
    panel_summary: list[tuple[str, dict[str, float]]] = []
    for name in strategies:
        per_sym: dict[str, dict[str, float]] = {}
        for sym in SYMBOLS:
            if sym not in bars_by_sym:
                continue
            r = run_for(name, sym, bars_by_sym[sym])
            if r:
                per_sym[sym] = r

        if not per_sym:
            continue

        # Per-symbol detail.
        print(f"--- {name} ---")
        for sym in SYMBOLS:
            r = per_sym.get(sym)
            if r:
                tag = "✓" if r["pass"] else " "
                print(
                    f"  {tag} {sym:<6} sharpe={r['sharpe']:>+5.2f} (std {r['std']:>4.2f}) "
                    f"trades={r['trades']:>3} expect={r['expect']:>+8.1f} corr={r['corr']:>+5.2f}"
                )
            else:
                print(f"    {sym:<6} (no data)")

        sharpes = [r["sharpe"] for r in per_sym.values()]
        expects = [r["expect"] for r in per_sym.values()]
        trades = sum(r["trades"] for r in per_sym.values())
        n_pass = sum(1 for r in per_sym.values() if r["pass"])
        panel = {
            "panel_mean_sharpe": fmean(sharpes),
            "panel_std_sharpe": pstdev(sharpes) if len(sharpes) > 1 else 0.0,
            "n_pass_rubric": n_pass,
            "n_total": len(per_sym),
            "panel_mean_expect": fmean(expects),
            "panel_total_trades": trades,
        }
        rank_score = panel["panel_mean_sharpe"] / max(panel["panel_std_sharpe"], 0.1)
        panel["rank_score"] = rank_score
        panel_summary.append((name, panel))
        print(
            f"  panel: mean={panel['panel_mean_sharpe']:+.2f} "
            f"std={panel['panel_std_sharpe']:.2f} "
            f"pass={n_pass}/{len(per_sym)}  trades={trades}  "
            f"expect={panel['panel_mean_expect']:+.1f}  "
            f"rank_score={rank_score:+.2f}\n"
        )

    # Leaderboard.
    print("=" * 90)
    print(f"{'rank':>4} {'strategy':<26} {'mean':>7} {'std':>6} {'pass':>5} {'trades':>7} "
          f"{'expect':>8} {'rank_score':>11}")
    print("=" * 90)
    panel_summary.sort(key=lambda x: -x[1]["rank_score"])
    for i, (name, p) in enumerate(panel_summary, 1):
        print(f"{i:>4} {name:<26} {p['panel_mean_sharpe']:>+7.2f} {p['panel_std_sharpe']:>6.2f} "
              f"{p['n_pass_rubric']:>2}/{p['n_total']:<2}  "
              f"{p['panel_total_trades']:>7d} {p['panel_mean_expect']:>+8.1f} "
              f"{p['rank_score']:>+11.2f}")


if __name__ == "__main__":
    main()
