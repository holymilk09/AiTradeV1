"""Risk gate invariants."""

from __future__ import annotations

from pathlib import Path

from aitrade.execution.orders import OrderRequest, Side
from aitrade.execution.risk import RiskGate


def _order(qty: float = 10, side: Side = Side.BUY) -> OrderRequest:
    return OrderRequest(symbol="AAPL", side=side, qty=qty)


def _gate(**kwargs: object) -> RiskGate:
    base = {
        "max_position_usd": 5_000.0,
        "max_daily_loss_usd": 500.0,
        "max_orders_per_min": 5,
    }
    base.update(kwargs)
    return RiskGate(**base)  # type: ignore[arg-type]


def test_allows_within_caps() -> None:
    gate = _gate()
    d = gate.check(_order(qty=10), reference_price=100.0)
    assert d.allowed, d.reason


def test_blocks_when_position_notional_exceeds_cap() -> None:
    gate = _gate(max_position_usd=500.0)
    d = gate.check(_order(qty=10), reference_price=100.0)
    assert not d.allowed
    assert "exceeds cap" in d.reason


def test_blocks_after_daily_loss_breach() -> None:
    gate = _gate(max_daily_loss_usd=50.0)
    gate.record_realized_pnl(-51.0)
    d = gate.check(_order(), reference_price=100.0)
    assert not d.allowed
    assert "halted" in d.reason


def test_reset_day_clears_halt() -> None:
    gate = _gate(max_daily_loss_usd=50.0)
    gate.record_realized_pnl(-200.0)
    assert gate.halted
    gate.reset_day()
    assert not gate.halted


def test_rate_limits_after_max_per_min() -> None:
    gate = _gate(max_orders_per_min=2)
    for _ in range(2):
        d = gate.check(_order(), reference_price=100.0)
        assert d.allowed
        gate.commit()
    d = gate.check(_order(), reference_price=100.0)
    assert not d.allowed
    assert "rate limit" in d.reason


def test_kill_switch_file_blocks(tmp_path: Path) -> None:
    switch = tmp_path / "KILL"
    switch.write_text("halt")
    gate = _gate(kill_switch_path=switch)
    d = gate.check(_order(), reference_price=100.0)
    assert not d.allowed
    assert "kill switch" in d.reason


def test_non_positive_price_rejected() -> None:
    gate = _gate()
    d = gate.check(_order(), reference_price=0)
    assert not d.allowed
