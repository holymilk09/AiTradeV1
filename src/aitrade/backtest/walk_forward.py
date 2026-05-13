"""Walk-forward backtesting.

Rolling train/test windows over a single bar series, returning per-fold
metrics + aggregate. A strategy with edge produces stable fold Sharpes;
a noisy strategy produces a wide fold-Sharpe distribution that
straddles zero. The aggregate hides this — folds expose it.

Training-fold metrics are computed but not used to refit anything here;
the harness is regime-agnostic. Param fitting happens upstream (caller
re-instantiates the strategy per fold with chosen params).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from aitrade.backtest.metrics import BacktestMetrics
from aitrade.backtest.simple_runner import BacktestConfig, BacktestResult, run_backtest
from aitrade.data.models import Bar
from aitrade.strategy.base import Strategy

StrategyFactory = Callable[[], Strategy]


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_result: BacktestResult
    test_result: BacktestResult


@dataclass(frozen=True, slots=True)
class WalkForwardSummary:
    folds: list[Fold]
    test_sharpe_mean: float
    test_sharpe_std: float
    test_expectancy_mean: float
    test_profit_factor_median: float
    test_max_dd_pct_worst: float
    test_trade_count_total: int
    train_test_sharpe_corr: float  # >0 means train Sharpe predicts test Sharpe


def _slice(bars: Sequence[Bar], start: pd.Timestamp, end: pd.Timestamp) -> list[Bar]:
    return [b for b in bars if start <= pd.Timestamp(b.timestamp) < end]


def walk_forward(
    factory: StrategyFactory,
    bars: Sequence[Bar],
    *,
    train: pd.Timedelta,
    test: pd.Timedelta,
    step: pd.Timedelta | None = None,
    config: BacktestConfig | None = None,
) -> WalkForwardSummary:
    """Run a strategy through a rolling walk-forward.

    Args:
        factory: zero-arg callable returning a fresh Strategy instance.
            Called per fold so state doesn't leak between folds.
        bars: chronologically sorted bars (single or multi-symbol).
        train: training-window duration.
        test: test-window duration.
        step: roll size; default = test (non-overlapping test windows).
        config: backtest config.
    """
    if not bars:
        raise ValueError("no bars")
    cfg = config or BacktestConfig()
    step = step or test

    ts = pd.Series([pd.Timestamp(b.timestamp) for b in bars]).sort_values()
    series_start = ts.iloc[0]
    series_end = ts.iloc[-1]

    folds: list[Fold] = []
    cursor = series_start
    idx = 0
    while cursor + train + test <= series_end:
        train_start = cursor
        train_end = cursor + train
        test_start = train_end
        test_end = test_start + test

        train_bars = _slice(bars, train_start, train_end)
        test_bars = _slice(bars, test_start, test_end)
        if len(train_bars) < 2 or len(test_bars) < 2:
            cursor += step
            idx += 1
            continue

        train_result = run_backtest(factory(), train_bars, cfg)
        test_result = run_backtest(factory(), test_bars, cfg)
        folds.append(
            Fold(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_result=train_result,
                test_result=test_result,
            )
        )
        cursor += step
        idx += 1

    if not folds:
        raise ValueError(
            "walk_forward produced 0 folds; check train/test sizes vs series length"
        )

    test_metrics: list[BacktestMetrics] = [f.test_result.metrics for f in folds]
    test_sharpes = [m.sharpe for m in test_metrics]
    train_sharpes = [f.train_result.metrics.sharpe for f in folds]
    finite_pfs = [m.profit_factor for m in test_metrics if m.profit_factor != float("inf")]
    corr_raw = (
        float(pd.Series(train_sharpes).corr(pd.Series(test_sharpes)))
        if len(folds) >= 3
        else 0.0
    )
    corr = 0.0 if math.isnan(corr_raw) else corr_raw
    return WalkForwardSummary(
        folds=folds,
        test_sharpe_mean=statistics.fmean(test_sharpes),
        test_sharpe_std=statistics.pstdev(test_sharpes) if len(test_sharpes) > 1 else 0.0,
        test_expectancy_mean=statistics.fmean(m.expectancy for m in test_metrics),
        test_profit_factor_median=statistics.median(finite_pfs) if finite_pfs else 0.0,
        test_max_dd_pct_worst=min(m.max_drawdown_pct for m in test_metrics),
        test_trade_count_total=sum(m.num_trades for m in test_metrics),
        train_test_sharpe_corr=corr,
    )


def summary_to_dataframe(summary: WalkForwardSummary) -> pd.DataFrame:
    """Per-fold metrics as a DataFrame for notebooks / parquet export."""
    rows = []
    for f in summary.folds:
        tm = f.test_result.metrics
        rows.append(
            {
                "fold": f.index,
                "train_start": f.train_start,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "test_sharpe": tm.sharpe,
                "test_sortino": tm.sortino,
                "test_cagr_pct": tm.cagr_pct,
                "test_max_dd_pct": tm.max_drawdown_pct,
                "test_mar": tm.mar,
                "test_trades": tm.num_trades,
                "test_hit_rate": tm.hit_rate,
                "test_profit_factor": tm.profit_factor,
                "test_expectancy": tm.expectancy,
                "train_sharpe": f.train_result.metrics.sharpe,
            }
        )
    return pd.DataFrame(rows)
