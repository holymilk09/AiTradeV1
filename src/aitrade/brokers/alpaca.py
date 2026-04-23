"""Alpaca adapter implementing BrokerClient.

The factory ``build_client`` is the *only* sanctioned way to construct this.
It enforces the paper-by-default guardrail — see ``aitrade.config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from aitrade.brokers.base import AccountSummary, BrokerClient, Position
from aitrade.config import Settings, assert_paper_or_unlocked, get_settings
from aitrade.execution.orders import (
    OrderAck,
    OrderRequest,
    OrderStatus,
    OrderType,
    Side,
)

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


_STATUS_MAP = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


class AlpacaBroker(BrokerClient):
    def __init__(self, trading_client: TradingClient, *, is_paper: bool) -> None:
        self._client = trading_client
        self._is_paper = is_paper

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def get_account(self) -> AccountSummary:
        acct = self._client.get_account()
        return AccountSummary(
            account_id=str(acct.id),
            cash=float(acct.cash),
            equity=float(acct.equity),
            buying_power=float(acct.buying_power),
            currency=getattr(acct, "currency", "USD") or "USD",
            is_paper=self._is_paper,
        )

    def get_positions(self) -> list[Position]:
        return [
            Position(
                symbol=str(p.symbol),
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
            )
            for p in self._client.get_all_positions()
        ]

    def submit_order(self, order: OrderRequest) -> OrderAck:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = AlpacaSide.BUY if order.side is Side.BUY else AlpacaSide.SELL
        tif = AlpacaTIF(order.time_in_force.value)

        if order.order_type is OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=order.client_order_id,
            )
        else:
            if order.limit_price is None:
                raise ValueError("limit_price required for LIMIT orders")
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )

        submitted = self._client.submit_order(order_data=req)
        status = _STATUS_MAP.get(str(submitted.status).lower(), OrderStatus.NEW)
        logger.info(
            "order submitted symbol={} side={} qty={} type={} status={} id={}",
            order.symbol,
            order.side.value,
            order.qty,
            order.order_type.value,
            status.value,
            submitted.id,
        )
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=str(submitted.id),
            status=status,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        self._client.cancel_order_by_id(broker_order_id)
        logger.info("cancel order id={}", broker_order_id)

    def cancel_all(self) -> int:
        responses = self._client.cancel_orders()
        logger.info("cancel_all count={}", len(responses))
        return len(responses)


def build_client(
    *,
    confirm_live: bool = False,
    settings: Settings | None = None,
) -> AlpacaBroker:
    """Construct an Alpaca broker, honoring the paper-by-default guardrail.

    Paper mode is always safe. Live mode requires BOTH:
      - env var ``ALPACA_LIVE_TRADE=true``
      - ``confirm_live=True`` at this call site
    See ``aitrade.config.assert_paper_or_unlocked``.
    """
    from alpaca.trading.client import TradingClient

    s = settings or get_settings()
    if not s.has_credentials:
        raise RuntimeError(
            "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."
        )

    live = assert_paper_or_unlocked(confirm_live=confirm_live, settings=s)
    paper = not live

    client = TradingClient(
        api_key=s.alpaca_api_key.get_secret_value(),
        secret_key=s.alpaca_secret_key.get_secret_value(),
        paper=paper,
    )
    logger.info("alpaca broker ready mode={}", "LIVE" if live else "paper")
    return AlpacaBroker(client, is_paper=paper)
