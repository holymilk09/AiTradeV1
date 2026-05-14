"""Phase 5 dynamic exit planning — ATR-based stop/target with VIX widening."""

from __future__ import annotations

import pytest

from aitrade.execution.exits import StopPlan, compute_stop_plan
from aitrade.execution.orders import Side


def test_long_plan_stop_below_target_above_entry() -> None:
    plan = compute_stop_plan(
        entry_price=100.0,
        atr=2.0,
        side=Side.BUY,
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
        vix=15.0,
    )
    assert plan is not None
    assert plan.stop_loss_price < 100.0
    assert plan.take_profit_price > 100.0
    # 2x stop ATR distance: 2.0 * 1.5 = 3.0 → stop at 97.0
    assert plan.stop_loss_price == pytest.approx(97.0)
    # 2x target ATR distance: 2.0 * 3.0 = 6.0 → target at 106.0
    assert plan.take_profit_price == pytest.approx(106.0)


def test_short_plan_stop_above_target_below_entry() -> None:
    plan = compute_stop_plan(
        entry_price=100.0,
        atr=2.0,
        side=Side.SELL,
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
        vix=15.0,
    )
    assert plan is not None
    assert plan.stop_loss_price > 100.0
    assert plan.take_profit_price < 100.0


def test_high_vix_widens_stop_distance() -> None:
    """Stop sits further out when VIX is elevated — same trade in noisier weather."""
    calm = compute_stop_plan(
        entry_price=100.0, atr=2.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
    )
    spicy = compute_stop_plan(
        entry_price=100.0, atr=2.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=30.0,
    )
    assert calm is not None
    assert spicy is not None
    calm_dist = 100.0 - calm.stop_loss_price
    spicy_dist = 100.0 - spicy.stop_loss_price
    assert spicy_dist > calm_dist
    # Reward:risk preserved — both legs widen by the same factor.
    assert calm.reward_risk_ratio(100.0, Side.BUY) == pytest.approx(
        spicy.reward_risk_ratio(100.0, Side.BUY), abs=0.01
    )


def test_min_stop_pct_protects_against_degenerate_atr() -> None:
    """A near-zero ATR (illiquid name) gets a 0.5% floor on stop distance."""
    plan = compute_stop_plan(
        entry_price=100.0,
        atr=0.001,  # collapsed ATR
        side=Side.BUY,
        stop_atr_mult=1.5,
        target_atr_mult=3.0,
        vix=15.0,
    )
    assert plan is not None
    # Floor: 0.5% of 100 = 0.5 distance → stop at 99.5 (or similar)
    assert plan.stop_loss_price <= 99.5


def test_rejects_bad_reward_risk_ratio() -> None:
    """compute_stop_plan returns None when R:R is structurally below 1.5."""
    plan = compute_stop_plan(
        entry_price=100.0,
        atr=2.0,
        side=Side.BUY,
        stop_atr_mult=2.0,
        target_atr_mult=2.0,  # R:R = 1.0 → too low
        vix=15.0,
    )
    assert plan is None


def test_rejects_zero_or_negative_atr() -> None:
    assert compute_stop_plan(
        entry_price=100.0, atr=0.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
    ) is None
    assert compute_stop_plan(
        entry_price=100.0, atr=-0.5, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
    ) is None


def test_rejects_zero_or_negative_entry() -> None:
    assert compute_stop_plan(
        entry_price=0.0, atr=2.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
    ) is None


def test_trail_percent_set_when_trailing_enabled() -> None:
    plan = compute_stop_plan(
        entry_price=100.0, atr=2.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
        enable_trailing=True,
    )
    assert plan is not None
    assert plan.trail_percent > 0.0


def test_trail_percent_zero_when_disabled() -> None:
    plan = compute_stop_plan(
        entry_price=100.0, atr=2.0, side=Side.BUY,
        stop_atr_mult=1.5, target_atr_mult=3.0, vix=15.0,
        enable_trailing=False,
    )
    assert plan is not None
    assert plan.trail_percent == 0.0


def test_reward_risk_ratio_long_and_short() -> None:
    long_plan = StopPlan(stop_loss_price=95.0, take_profit_price=110.0)
    assert long_plan.reward_risk_ratio(100.0, Side.BUY) == pytest.approx(2.0)
    short_plan = StopPlan(stop_loss_price=105.0, take_profit_price=90.0)
    assert short_plan.reward_risk_ratio(100.0, Side.SELL) == pytest.approx(2.0)
