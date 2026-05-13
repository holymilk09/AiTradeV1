"""Side-by-side comparison of strategy runs.

The output is one DataFrame row per (label, fold-or-aggregate). Drop it
into a notebook and sort by whichever metric you trust most for the
current decision. Designed so M2 'pick the survivors' is mechanical, not
vibes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from aitrade.backtest.simple_runner import BacktestConfig
from aitrade.backtest.walk_forward import (
    StrategyFactory,
    WalkForwardSummary,
    summary_to_dataframe,
    walk_forward,
)
from aitrade.data.models import Bar


@dataclass(frozen=True, slots=True)
class Run:
    label: str
    factory: StrategyFactory


def run_bench(
    runs: Sequence[Run],
    bars: Sequence[Bar],
    *,
    train: pd.Timedelta,
    test: pd.Timedelta,
    step: pd.Timedelta | None = None,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run each strategy through walk-forward, return (per-fold, summary) frames.

    The summary frame is the leaderboard — one row per label, columns
    are the aggregate stats from WalkForwardSummary. The per-fold frame
    is for diagnostic plots.
    """
    fold_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for r in runs:
        summary = walk_forward(
            r.factory, bars, train=train, test=test, step=step, config=config
        )
        per_fold = summary_to_dataframe(summary)
        per_fold.insert(0, "label", r.label)
        fold_frames.append(per_fold)
        summaries.append(
            {
                "label": r.label,
                "folds": len(summary.folds),
                "sharpe_mean": summary.test_sharpe_mean,
                "sharpe_std": summary.test_sharpe_std,
                "expectancy_mean": summary.test_expectancy_mean,
                "profit_factor_median": summary.test_profit_factor_median,
                "max_dd_worst_pct": summary.test_max_dd_pct_worst,
                "trades_total": summary.test_trade_count_total,
                "train_test_sharpe_corr": _safe_corr(summary),
            }
        )
    return pd.concat(fold_frames, ignore_index=True), pd.DataFrame(summaries)


def _safe_corr(summary: WalkForwardSummary) -> float:
    c = summary.train_test_sharpe_corr
    return 0.0 if math.isnan(c) else c
