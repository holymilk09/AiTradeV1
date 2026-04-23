"""Simple backtest runner over synthetic bars produces a valid equity curve + metrics."""

from __future__ import annotations

from aitrade.backtest.simple_runner import BacktestConfig, run_backtest
from aitrade.data.models import Bar
from aitrade.strategy.examples.sma_crossover import SmaCrossover


def test_backtest_end_to_end(synthetic_bars: list[Bar]) -> None:
    strat = SmaCrossover(symbol="TEST", fast=5, slow=20)
    cfg = BacktestConfig(starting_cash=100_000.0, target_position_notional=10_000.0)
    result = run_backtest(strat, synthetic_bars, cfg)

    assert not result.equity_curve.empty
    assert result.equity_curve.iloc[0] == cfg.starting_cash
    # Ramping price should produce at least one fill.
    assert len(result.fills) >= 1
    # Metrics should be finite.
    assert result.metrics.final_equity > 0
