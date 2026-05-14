"""Walk-forward harness produces folds + summary stats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from aitrade.backtest.compare import Run, run_bench
from aitrade.backtest.metrics import compute_metrics, periods_per_year
from aitrade.backtest.simple_runner import BacktestConfig, run_backtest
from aitrade.backtest.walk_forward import summary_to_dataframe, walk_forward
from aitrade.data.models import Bar, Timeframe
from aitrade.strategy.examples.sma_crossover import SmaCrossover


def _trend_bars(n: int = 400, slope: float = 0.05) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i in range(n):
        # Half flat, half trend: forces signal in second half.
        price = 100.0 if i < n // 2 else 100.0 + (i - n // 2) * slope
        bars.append(
            Bar(
                symbol="TEST",
                timestamp=start + timedelta(days=i),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1_000_000,
            )
        )
    return bars


def test_walk_forward_produces_folds() -> None:
    bars = _trend_bars(n=400)
    summary = walk_forward(
        lambda: SmaCrossover(symbol="TEST", fast=5, slow=20),
        bars,
        train=pd.Timedelta(days=120),
        test=pd.Timedelta(days=60),
    )
    assert len(summary.folds) >= 3
    assert summary.test_sharpe_std >= 0.0
    df = summary_to_dataframe(summary)
    assert {"fold", "test_sharpe", "test_trades"}.issubset(df.columns)
    assert len(df) == len(summary.folds)


def test_walk_forward_raises_on_too_little_data() -> None:
    bars = _trend_bars(n=30)
    with pytest.raises(ValueError, match="0 folds"):
        walk_forward(
            lambda: SmaCrossover(symbol="TEST", fast=5, slow=20),
            bars,
            train=pd.Timedelta(days=120),
            test=pd.Timedelta(days=60),
        )


def test_run_bench_returns_leaderboard() -> None:
    bars = _trend_bars(n=400)
    runs = [
        Run("sma_5_20", lambda: SmaCrossover(symbol="TEST", fast=5, slow=20)),
        Run("sma_10_30", lambda: SmaCrossover(symbol="TEST", fast=10, slow=30)),
    ]
    per_fold, leaderboard = run_bench(
        runs, bars, train=pd.Timedelta(days=120), test=pd.Timedelta(days=60)
    )
    assert set(leaderboard["label"]) == {"sma_5_20", "sma_10_30"}
    assert {"sharpe_mean", "sharpe_std", "trades_total"}.issubset(leaderboard.columns)
    assert per_fold["label"].nunique() == 2


def test_metrics_have_new_fields() -> None:
    bars = _trend_bars(n=200)
    result = run_backtest(
        SmaCrossover(symbol="TEST", fast=5, slow=20),
        bars,
        BacktestConfig(),
    )
    m = result.metrics
    # Existing fields preserved.
    assert hasattr(m, "sharpe")
    assert hasattr(m, "max_drawdown_pct")
    # New fields.
    assert hasattr(m, "profit_factor")
    assert hasattr(m, "expectancy")
    assert hasattr(m, "mar")
    assert hasattr(m, "longest_losing_streak")
    assert hasattr(m, "return_skew")
    assert m.longest_losing_streak >= 0


def test_periods_per_year_table() -> None:
    assert periods_per_year(Timeframe.DAY_1) == 252
    assert periods_per_year(Timeframe.HOUR_1) == 252 * 7
    assert periods_per_year(Timeframe.MIN_1) == 252 * 390


def test_compute_metrics_empty() -> None:
    m = compute_metrics(pd.Series(dtype=float), [])
    assert m.num_trades == 0
    assert m.sharpe == 0.0
