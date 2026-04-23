"""Order router: risk gate → broker → trade journal."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from aitrade.brokers.base import BrokerClient
from aitrade.execution.orders import OrderAck, OrderRequest
from aitrade.execution.risk import RiskGate


@dataclass(frozen=True, slots=True)
class SubmitResult:
    accepted: bool
    reason: str = ""
    ack: OrderAck | None = None


class Executor:
    def __init__(self, broker: BrokerClient, risk: RiskGate) -> None:
        self._broker = broker
        self._risk = risk

    @property
    def risk(self) -> RiskGate:
        return self._risk

    def submit(
        self,
        order: OrderRequest,
        *,
        reference_price: float,
        current_position_qty: float = 0.0,
    ) -> SubmitResult:
        decision = self._risk.check(
            order,
            reference_price=reference_price,
            current_position_qty=current_position_qty,
        )
        if not decision.allowed:
            logger.warning(
                "order blocked symbol={} side={} qty={} reason={}",
                order.symbol,
                order.side.value,
                order.qty,
                decision.reason,
            )
            return SubmitResult(False, decision.reason)

        ack = self._broker.submit_order(order)
        self._risk.commit()
        return SubmitResult(True, ack=ack)
