"""Lightweight bar-by-bar backtest runner.

Used by tests + as a fast sanity-check. For production / research-grade
backtests with realistic queue/slippage modeling, use ``nautilus_runner``.

Slippage model: linear on price (bps of close). Fees: bps of notional.
Fills at next bar's open to avoid lookahead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger

from aitrade.backtest.metrics import BacktestMetrics, compute_metrics
from aitrade.data.models import Bar
from aitrade.strategy.base import Strategy
from aitrade.strategy.signal import Direction


@dataclass
class BacktestConfig:
    starting_cash: float = 100_000.0
    slippage_bps: float = 2.0
    fee_bps: float = 1.0
    target_position_notional: float = 10_000.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trade_pnls: list[float]
    metrics: BacktestMetrics
    fills: list[tuple[datetime, str, str, float, float]] = field(default_factory=list)
    # (ts, symbol, side, qty, price)


def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    mult = 1.0 + (slippage_bps / 10_000.0) * (1.0 if side == "buy" else -1.0)
    return price * mult


def run_backtest(
    strategy: Strategy,
    bars: Iterable[Bar],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    bar_list = list(bars)
    if not bar_list:
        raise ValueError("no bars to backtest")

    cash = cfg.starting_cash
    qty_held: dict[str, float] = {}
    avg_cost: dict[str, float] = {}
    pending_target: dict[str, Direction] = {}
    equity_series: dict[datetime, float] = {}
    trade_pnls: list[float] = []
    fills: list[tuple[datetime, str, str, float, float]] = []

    for i, bar in enumerate(bar_list):
        # Execute previous bar's target at this bar's open (no lookahead).
        target = pending_target.pop(bar.symbol, None)
        if target is not None:
            held = qty_held.get(bar.symbol, 0.0)
            desired_qty = 0.0
            if target is Direction.LONG:
                desired_qty = max(1.0, cfg.target_position_notional // bar.open)
            delta = desired_qty - held
            if abs(delta) >= 1:
                side = "buy" if delta > 0 else "sell"
                px = _apply_slippage(bar.open, side, cfg.slippage_bps)
                notional = abs(delta) * px
                fee = notional * (cfg.fee_bps / 10_000.0)
                if side == "buy":
                    cash -= notional + fee
                    new_held = held + delta
                    if new_held > 0:
                        avg_cost[bar.symbol] = (
                            (avg_cost.get(bar.symbol, 0.0) * held + px * delta) / new_held
                        )
                    qty_held[bar.symbol] = new_held
                else:
                    cash += notional - fee
                    # Realized PnL on the closed portion
                    closed_qty = min(abs(delta), held) if held > 0 else 0
                    if closed_qty > 0:
                        pnl = (px - avg_cost.get(bar.symbol, px)) * closed_qty - fee
                        trade_pnls.append(pnl)
                    qty_held[bar.symbol] = held + delta
                fills.append((bar.timestamp, bar.symbol, side, abs(delta), px))

        # Dispatch bar to strategy for next-bar decision.
        signal = strategy.on_bar(bar)
        if signal is not None:
            pending_target[signal.symbol] = signal.direction

        # Mark-to-market equity.
        position_value = sum(
            q * bar.close for sym, q in qty_held.items() if sym == bar.symbol
        )
        # For multi-symbol, would need current close per symbol; single-symbol OK here.
        equity_series[bar.timestamp] = cash + position_value

        if i % 500 == 0:
            logger.debug(
                "[{}/{}] cash={:.2f} held={} equity={:.2f}",
                i + 1,
                len(bar_list),
                cash,
                qty_held,
                equity_series[bar.timestamp],
            )

    equity_curve = pd.Series(equity_series).sort_index()
    metrics = compute_metrics(equity_curve, trade_pnls)
    return BacktestResult(
        equity_curve=equity_curve,
        trade_pnls=trade_pnls,
        metrics=metrics,
        fills=fills,
    )
