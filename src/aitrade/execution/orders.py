"""Broker-agnostic order DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    TRAILING_STOP = "trailing_stop"


class OrderClass(str, Enum):
    """Multi-leg order topology (Alpaca terminology).

    ``SIMPLE`` is a single-leg order. ``BRACKET`` attaches a stop-loss + a
    take-profit child to the parent so the broker manages exits server-side
    even if our process dies. ``OTO`` ("one triggers other") is used for a
    parent + take-profit only; ``OCO`` for a stop-loss + take-profit pair on
    an existing position.
    """

    SIMPLE = "simple"
    BRACKET = "bracket"
    OTO = "oto"
    OCO = "oco"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(str, Enum):
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """A single broker-agnostic order request.

    Bracket orders attach a stop_loss + take_profit pair to the parent. The
    broker (Alpaca) manages those legs server-side so an exit fires even if
    this process crashes — critical for unattended overnight runs.

    Trailing stops follow price by ``trail_percent`` (or ``trail_price``
    absolute distance) and replace a static stop once the trade is in the
    money. Either trail field is mutually exclusive with the other.
    """

    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: str = field(default_factory=lambda: f"aitrade-{uuid4().hex[:16]}")
    strategy_id: str = "unknown"
    # Bracket / OCO topology — informational unless order_class is set.
    order_class: OrderClass = OrderClass.SIMPLE
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    # Trailing-stop (used when order_type == TRAILING_STOP). Pick one.
    trail_percent: float | None = None
    trail_price: float | None = None


@dataclass(frozen=True, slots=True)
class OrderAck:
    client_order_id: str
    broker_order_id: str
    status: OrderStatus
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Fill:
    client_order_id: str
    broker_order_id: str
    symbol: str
    side: Side
    qty: float
    price: float
    fee: float = 0.0
    filled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
