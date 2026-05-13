"""Reasoner value test (M4).

Decide whether the LLM reasoner adds measurable edge over the strategy
+ conviction composite alone. Compares two trade tables:

- A: trades the strategy would take (every setup row).
- B: trades the strategy would take AND the reasoner accepts.

For each side: expectancy, win rate, profit factor, max forward DD,
trade count. A reasoner is "valuable" iff B beats A net of cost AND
trade count remains statistically meaningful.

The harness is agnostic to the accept function — pass a mock for tests,
a real ``Reasoner``-backed function for offline replay against
historical setups.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

AcceptFn = Callable[[pd.Series], bool]


@dataclass(frozen=True, slots=True)
class SideMetrics:
    n_trades: int
    win_rate: float
    expectancy: float  # mean forward return per trade
    profit_factor: float  # sum(positive fwd) / |sum(negative fwd)|
    worst_trade: float
    best_trade: float
    median_trade: float


@dataclass(frozen=True, slots=True)
class ValueTestResult:
    side_a: SideMetrics  # all setups
    side_b: SideMetrics  # accepted setups
    acceptance_rate: float
    expectancy_delta: float  # B - A
    profit_factor_delta: float  # B - A
    cost_per_trade: float  # passed through for net-of-cost decisions
    net_expectancy_b: float  # side_b.expectancy - cost_per_trade


def _side_metrics(returns: pd.Series) -> SideMetrics:
    if returns.empty:
        return SideMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_loss = float(losses.abs().sum())
    profit_factor = float(wins.sum()) / gross_loss if gross_loss > 0 else (
        float("inf") if not wins.empty else 0.0
    )
    return SideMetrics(
        n_trades=int(len(returns)),
        win_rate=float(len(wins) / len(returns)),
        expectancy=float(returns.mean()),
        profit_factor=profit_factor,
        worst_trade=float(returns.min()),
        best_trade=float(returns.max()),
        median_trade=float(returns.median()),
    )


def value_test(
    setups: pd.DataFrame,
    accept_fn: AcceptFn,
    *,
    return_col: str = "fwd_return",
    cost_per_trade: float = 0.0,
) -> ValueTestResult:
    """Run the comparison and return the result.

    Args:
        setups: rows produced by ``signals.calibration.collect_setups``.
        accept_fn: row -> bool. True means the reasoner would take the trade.
        return_col: name of the forward-return column on ``setups``.
        cost_per_trade: token-cost / fee per accepted trade, in return units.
            Used to compute ``net_expectancy_b`` for the keep/cut decision.
    """
    if return_col not in setups.columns:
        raise KeyError(f"setups missing '{return_col}' column")

    if setups.empty:
        empty = _side_metrics(pd.Series(dtype=float))
        return ValueTestResult(
            side_a=empty,
            side_b=empty,
            acceptance_rate=0.0,
            expectancy_delta=0.0,
            profit_factor_delta=0.0,
            cost_per_trade=cost_per_trade,
            net_expectancy_b=-cost_per_trade,
        )

    accept_mask = setups.apply(accept_fn, axis=1)
    accept_mask = accept_mask.astype(bool)
    returns_a = setups[return_col]
    returns_b = setups.loc[accept_mask, return_col]

    a = _side_metrics(returns_a)
    b = _side_metrics(returns_b)
    return ValueTestResult(
        side_a=a,
        side_b=b,
        acceptance_rate=float(accept_mask.mean()) if len(accept_mask) else 0.0,
        expectancy_delta=b.expectancy - a.expectancy,
        profit_factor_delta=b.profit_factor - a.profit_factor,
        cost_per_trade=cost_per_trade,
        net_expectancy_b=b.expectancy - cost_per_trade,
    )


def decision_string(result: ValueTestResult, *, min_trades: int = 50) -> str:
    """Human-readable keep/cut recommendation.

    Rules:
    - Net B expectancy must exceed A's expectancy.
    - B must have at least ``min_trades`` to be statistically meaningful.
    - B's profit factor must not regress badly.
    """
    if result.side_b.n_trades < min_trades:
        return (
            f"INSUFFICIENT_DATA: only {result.side_b.n_trades} accepted trades; "
            f"need >= {min_trades}. KEEP for now, gather more."
        )
    if result.net_expectancy_b <= result.side_a.expectancy:
        return (
            f"CUT: net B expectancy {result.net_expectancy_b:.4f} does not exceed "
            f"A expectancy {result.side_a.expectancy:.4f}."
        )
    if result.profit_factor_delta < -0.5:
        return (
            f"CUT: profit factor regression {result.profit_factor_delta:+.2f}; "
            f"reasoner is filtering profitable trades."
        )
    return (
        f"KEEP: net B expectancy {result.net_expectancy_b:.4f} > A {result.side_a.expectancy:.4f}; "
        f"PF delta {result.profit_factor_delta:+.2f}; n_b={result.side_b.n_trades}."
    )
