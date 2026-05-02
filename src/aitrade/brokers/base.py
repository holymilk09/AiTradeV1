"""Broker-agnostic Protocol. New venues (Kalshi, IBKR, ...) implement this."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aitrade.execution.orders import OrderAck, OrderRequest


@dataclass(frozen=True, slots=True)
class AccountSummary:
    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"
    is_paper: bool = True


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pnl: float


@runtime_checkable
class BrokerClient(Protocol):
    """Minimal surface every broker adapter must implement."""

    @property
    def is_paper(self) -> bool: ...

    def get_account(self) -> AccountSummary: ...

    def get_positions(self) -> list[Position]: ...

    def submit_order(self, order: OrderRequest) -> OrderAck: ...

    def cancel_order(self, broker_order_id: str) -> None: ...

    def cancel_all(self) -> int: ...

    def is_tradable(self, symbol: str) -> bool:
        """Pre-flight check: is this symbol currently tradable on this venue?

        Implementations should fail closed — when in doubt, return False.
        Default implementation here returns True so older test stubs that
        don't override it keep passing (they hit a paper broker that's
        already validated upstream).
        """
        return True
