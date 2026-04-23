"""Performance metrics for an equity curve."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    total_return_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    num_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    final_equity: float


def compute_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    *,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    if equity_curve.empty:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)

    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    total_return = (end / start - 1.0) * 100 if start > 0 else 0.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = math.sqrt(periods_per_year) * float(returns.mean() / returns.std())
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    sortino = (
        math.sqrt(periods_per_year) * float(returns.mean() / downside.std())
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min()) * 100 if not drawdown.empty else 0.0

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    hit_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return BacktestMetrics(
        total_return_pct=total_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        num_trades=len(trade_pnls),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        final_equity=end,
    )
