"""Performance metrics for an equity curve + trade ledger.

The harness's job is to tell you clearly whether a strategy has edge.
That means going beyond Sharpe — Sharpe alone hides win/loss asymmetry,
fat tails, and drawdown depth. Every field here exists because it answers
a question Sharpe cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from aitrade.data.models import Timeframe

# Trading periods per year by bar timeframe. RTH equities convention.
_PERIODS_PER_YEAR: dict[Timeframe, int] = {
    Timeframe.MIN_1: 252 * 390,
    Timeframe.MIN_5: 252 * 78,
    Timeframe.MIN_15: 252 * 26,
    Timeframe.HOUR_1: 252 * 7,
    Timeframe.DAY_1: 252,
}


def periods_per_year(tf: Timeframe) -> int:
    return _PERIODS_PER_YEAR[tf]


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    # Equity-curve metrics
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    mar: float  # CAGR / |max DD|; >1 means recovers DD within a year of compounding
    # Trade-ledger metrics
    num_trades: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    median_trade: float
    profit_factor: float  # sum(wins) / |sum(losses)|; inf if no losses
    expectancy: float  # avg trade $ P&L
    longest_losing_streak: int
    # Return distribution
    return_skew: float
    return_kurtosis: float  # excess kurtosis
    # Bookkeeping
    final_equity: float


def _streak(pnls: list[float]) -> int:
    longest = current = 0
    for p in pnls:
        if p < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def compute_metrics(
    equity_curve: pd.Series,
    trade_pnls: list[float],
    *,
    timeframe: Timeframe = Timeframe.DAY_1,
) -> BacktestMetrics:
    if equity_curve.empty:
        return BacktestMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            mar=0.0,
            num_trades=0,
            hit_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            median_trade=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            longest_losing_streak=0,
            return_skew=0.0,
            return_kurtosis=0.0,
            final_equity=0.0,
        )

    ppy = periods_per_year(timeframe)
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    total_return = (end / start - 1.0) * 100 if start > 0 else 0.0

    n_periods = max(len(equity_curve) - 1, 1)
    years = n_periods / ppy
    cagr = ((end / start) ** (1.0 / years) - 1.0) * 100 if years > 0 and start > 0 else 0.0

    returns = equity_curve.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = math.sqrt(ppy) * float(returns.mean() / returns.std())
    else:
        sharpe = 0.0
    downside = returns[returns < 0]
    sortino = (
        math.sqrt(ppy) * float(returns.mean() / downside.std())
        if len(downside) > 1 and downside.std() > 0
        else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd_pct = float(drawdown.min()) * 100 if not drawdown.empty else 0.0
    mar = cagr / abs(max_dd_pct) if max_dd_pct < 0 else 0.0

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    hit_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    median_trade = float(pd.Series(trade_pnls).median()) if trade_pnls else 0.0
    gross_loss = abs(sum(losses))
    profit_factor = (sum(wins) / gross_loss) if gross_loss > 0 else (math.inf if wins else 0.0)
    expectancy = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0

    skew = float(returns.skew()) if len(returns) > 2 else 0.0
    kurt = float(returns.kurt()) if len(returns) > 3 else 0.0

    return BacktestMetrics(
        total_return_pct=total_return,
        cagr_pct=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd_pct,
        mar=mar,
        num_trades=len(trade_pnls),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        median_trade=median_trade,
        profit_factor=profit_factor,
        expectancy=expectancy,
        longest_losing_streak=_streak(trade_pnls),
        return_skew=skew,
        return_kurtosis=kurt,
        final_equity=end,
    )
