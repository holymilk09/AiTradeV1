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
