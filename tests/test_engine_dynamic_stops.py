"""End-to-end glue: floor-trader decision → bracket order with ATR stop plan."""

from __future__ import annotations

import pytest

from aitrade.bots.engine_runner import _decision_to_order
from aitrade.execution.orders import OrderClass, OrderType, Side
from aitrade.reasoning.decision import FloorTraderDecision
from aitrade.strategy.signal import Direction


def _decision(
    *,
    pick: str = "AAPL",
    direction: Direction = Direction.LONG,
    stop_atr_mult: float | None = 1.5,
    target_atr_mult: float | None = 3.0,
    stop_price: float | None = None,
    target_price: float | None = None,
) -> FloorTraderDecision:
    return FloorTraderDecision(
        should_trade=True,
        pick_symbol=pick,
        direction=direction,
        target_notional_usd=2_000.0,
        confidence=0.75,
        thesis="Strong trend + buzz.",
        catalyst="Earnings beat.",
        entry_price=100.0,
        stop_price=stop_price,
        target_price=target_price,
        stop_atr_mult=stop_atr_mult,
        target_atr_mult=target_atr_mult,
    )


def test_long_entry_with_atr_stops_returns_bracket_order() -> None:
    """Fresh long entry + ATR multipliers + valid daily ATR → bracket order."""
    result = _decision_to_order(
        _decision(),
        held_qty=0.0,
        reference_price=100.0,
        target_notional=2_000.0,
        daily_atr=2.0,
        vix=15.0,
    )
    assert result is not None
    order, plan = result
    assert order.symbol == "AAPL"
    assert order.side is Side.BUY
    assert order.order_type is OrderType.MARKET
    assert order.order_class is OrderClass.BRACKET
    assert plan is not None
    assert order.stop_loss_price == plan.stop_loss_price
    assert order.take_profit_price == plan.take_profit_price
    # Stop sits below entry, target above (long).
    assert order.stop_loss_price < 100.0
    assert order.take_profit_price > 100.0


def test_close_position_returns_simple_order_no_plan() -> None:
    """FLAT direction with held qty → plain SELL, no bracket needed."""
    result = _decision_to_order(
        _decision(direction=Direction.FLAT),
        held_qty=10.0,
        reference_price=100.0,
        target_notional=2_000.0,
        daily_atr=2.0,
        vix=15.0,
    )
    assert result is not None
    order, plan = result
    assert order.side is Side.SELL
    assert order.order_class is OrderClass.SIMPLE
    assert plan is None


def test_missing_atr_with_atr_only_decision_skips_trade() -> None:
    """If the floor-trader specified ATR multiples but ATR is unavailable,
    the engine refuses the trade rather than sending it unprotected."""
    result = _decision_to_order(
        _decision(),  # ATR-mode decision
        held_qty=0.0,
        reference_price=100.0,
        target_notional=2_000.0,
        daily_atr=None,  # missing
        vix=15.0,
    )
    assert result is None


def test_absolute_prices_win_over_atr_multipliers() -> None:
    """When the floor-trader gives BOTH absolute prices and ATR mults, the
    structural levels (absolute) win — those reflect explicit S/R levels."""
    result = _decision_to_order(
        _decision(stop_price=95.0, target_price=110.0),
        held_qty=0.0,
        reference_price=100.0,
        target_notional=2_000.0,
        daily_atr=2.0,
        vix=15.0,
    )
    assert result is not None
    order, plan = result
    assert plan is not None
    assert order.stop_loss_price == pytest.approx(95.0)
    assert order.take_profit_price == pytest.approx(110.0)


def test_decision_below_min_reward_risk_skips_trade() -> None:
    """A 1.0 R:R proposal (stop=target ATRs) gets refused at the planner."""
    result = _decision_to_order(
        _decision(stop_atr_mult=2.0, target_atr_mult=2.0),  # R:R = 1.0
        held_qty=0.0,
        reference_price=100.0,
        target_notional=2_000.0,
        daily_atr=2.0,
        vix=15.0,
    )
    assert result is None


def test_high_vix_widens_resulting_stop() -> None:
    """Same decision in a high-VIX regime → wider absolute stop distance."""
    calm = _decision_to_order(
        _decision(),
        held_qty=0.0, reference_price=100.0, target_notional=2_000.0,
        daily_atr=2.0, vix=15.0,
    )
    spicy = _decision_to_order(
        _decision(),
        held_qty=0.0, reference_price=100.0, target_notional=2_000.0,
        daily_atr=2.0, vix=30.0,
    )
    assert calm is not None
    assert spicy is not None
    assert (100.0 - spicy[0].stop_loss_price) > (100.0 - calm[0].stop_loss_price)


def test_short_direction_v1_returns_none() -> None:
    result = _decision_to_order(
        _decision(direction=Direction.SHORT),
        held_qty=0.0, reference_price=100.0, target_notional=2_000.0,
        daily_atr=2.0, vix=15.0,
    )
    assert result is None


def test_pass_decision_returns_none() -> None:
    decision = FloorTraderDecision(should_trade=False, reason_for_pass="weak setup")
    result = _decision_to_order(
        decision,
        held_qty=0.0, reference_price=100.0, target_notional=2_000.0,
        daily_atr=2.0, vix=15.0,
    )
    assert result is None


def test_decision_schema_round_trips_atr_mults() -> None:
    """The new ATR-mult fields survive a model_dump → model_validate roundtrip."""
    d = FloorTraderDecision(
        should_trade=True,
        pick_symbol="AAPL",
        direction=Direction.LONG,
        target_notional_usd=2_000.0,
        confidence=0.8,
        thesis="x",
        catalyst="y",
        stop_atr_mult=1.2,
        target_atr_mult=2.5,
    )
    dumped = d.model_dump()
    reread = FloorTraderDecision.model_validate(dumped)
    assert reread.stop_atr_mult == 1.2
    assert reread.target_atr_mult == 2.5
