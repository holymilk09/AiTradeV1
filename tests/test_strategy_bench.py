"""Bench: registered strategies all run through walk-forward without error
and produce a comparable leaderboard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from aitrade.backtest.compare import Run, run_bench
from aitrade.data.models import Bar
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.examples.donchian_breakout import DonchianBreakout
from aitrade.strategy.examples.sma_crossover import SmaCrossover


def _bars(n: int = 600) -> list[Bar]:
    """Bars with regime change at the midpoint: choppy then trending up.

    Designed so different strategies catch different regimes:
      - sma + donchian benefit from the trend
      - bollinger benefits from the chop
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[Bar] = []
    for i in range(n):
        price = (
            100.0 + ((i % 7) - 3) * 0.4  # mean-reverting chop
            if i < n // 2
            else 100.0 + (i - n // 2) * 0.15  # steady uptrend
        )
        out.append(
            Bar(
                symbol="TEST",
                timestamp=start + timedelta(days=i),
                open=price,
                high=price + 0.2,
                low=price - 0.2,
                close=price,
                volume=1_000_000,
            )
        )
    return out


def test_bench_all_three_strategies() -> None:
    bars = _bars()
    runs = [
        Run("sma", lambda: SmaCrossover(symbol="TEST", fast=5, slow=20)),
        Run("bollinger", lambda: BollingerReversion(symbol="TEST", period=20, num_std=2.0)),
        Run("donchian", lambda: DonchianBreakout(symbol="TEST", entry_period=20, exit_period=10)),
    ]
    per_fold, leaderboard = run_bench(
        runs, bars, train=pd.Timedelta(days=180), test=pd.Timedelta(days=90)
    )
    assert set(leaderboard["label"]) == {"sma", "bollinger", "donchian"}
    assert (leaderboard["folds"] > 0).all()
    # Leaderboard should expose all aggregate metrics for sorting.
    assert {
        "sharpe_mean",
        "sharpe_std",
        "expectancy_mean",
        "profit_factor_median",
        "max_dd_worst_pct",
        "trades_total",
    }.issubset(leaderboard.columns)
    # Per-fold should have one row per (label, fold).
    assert per_fold["label"].nunique() == 3
