"""Executor invokes broker only when the risk gate allows."""

from __future__ import annotations

from dataclasses import dataclass, field

from aitrade.brokers.base import AccountSummary, BrokerClient, Position
from aitrade.execution.executor import Executor
from aitrade.execution.orders import OrderAck, OrderRequest, OrderStatus, Side
from aitrade.execution.risk import RiskGate


@dataclass
class FakeBroker(BrokerClient):
    calls: list[OrderRequest] = field(default_factory=list)
    halted_symbols: set[str] = field(default_factory=set)
    raise_on_is_tradable: bool = False

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> AccountSummary:
        return AccountSummary(
            account_id="fake", cash=0, equity=0, buying_power=0, is_paper=True
        )

    def get_positions(self) -> list[Position]:
        return []

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self.calls.append(order)
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id="br-1",
            status=OrderStatus.ACCEPTED,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        pass

    def cancel_all(self) -> int:
        return 0

    def is_tradable(self, symbol: str) -> bool:
        if self.raise_on_is_tradable:
            raise RuntimeError("get_asset network failure")
        return symbol not in self.halted_symbols


def _gate() -> RiskGate:
    return RiskGate(
        max_position_usd=5_000.0,
        max_daily_loss_usd=500.0,
        max_orders_per_min=5,
    )


def test_accepted_order_reaches_broker() -> None:
    broker = FakeBroker()
    exe = Executor(broker, _gate())
    order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10)
    result = exe.submit(order, reference_price=100.0)
    assert result.accepted
    assert len(broker.calls) == 1
    assert result.ack is not None
    assert result.ack.status is OrderStatus.ACCEPTED


def test_risk_block_prevents_broker_call() -> None:
    broker = FakeBroker()
    gate = _gate()
    gate.record_realized_pnl(-9999)
    exe = Executor(broker, gate)
    order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10)
    result = exe.submit(order, reference_price=100.0)
    assert not result.accepted
    assert not broker.calls


def test_halted_symbol_short_circuits_before_risk_gate() -> None:
    """Phase 5: halt detection runs before risk so we never tick the rate
    limiter on a symbol the venue won't even accept."""
    broker = FakeBroker(halted_symbols={"AAPL"})
    gate = _gate()
    exe = Executor(broker, gate)
    order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10)
    result = exe.submit(order, reference_price=100.0)
    assert not result.accepted
    assert "not tradable" in result.reason
    assert not broker.calls


def test_is_tradable_failure_fails_closed() -> None:
    """get_asset network failure must NOT silently allow the order through."""
    broker = FakeBroker(raise_on_is_tradable=True)
    exe = Executor(broker, _gate())
    order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10)
    result = exe.submit(order, reference_price=100.0)
    assert not result.accepted
    assert not broker.calls


def test_non_halted_symbol_still_reaches_broker() -> None:
    broker = FakeBroker(halted_symbols={"GME"})  # AAPL is fine
    exe = Executor(broker, _gate())
    order = OrderRequest(symbol="AAPL", side=Side.BUY, qty=10)
    result = exe.submit(order, reference_price=100.0)
    assert result.accepted
    assert len(broker.calls) == 1
