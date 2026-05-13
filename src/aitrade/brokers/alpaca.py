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
    OrderClass,
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

    def is_tradable(self, symbol: str) -> bool:
        """Pre-flight halt / delisting check.

        Returns True only when the symbol is *currently* tradable — not
        halted, not delisted, not deactivated. False in any other state
        including network failure (fail-closed: when in doubt, don't trade).
        """
        try:
            asset = self._client.get_asset(symbol)
        except Exception as exc:
            logger.warning(
                "is_tradable get_asset failed symbol={} err={}; failing closed",
                symbol,
                exc,
            )
            return False
        tradable = bool(getattr(asset, "tradable", False))
        raw_status = getattr(asset, "status", "")
        # Alpaca's AssetStatus is an Enum whose .value is "active" / "inactive".
        # Plain str() yields "AssetStatus.ACTIVE", which would silently fail
        # the equality check below for every real symbol — that was the bug.
        status = str(getattr(raw_status, "value", raw_status)).lower()
        return tradable and status == "active"

    def submit_order(self, order: OrderRequest) -> OrderAck:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            TakeProfitRequest,
            TrailingStopOrderRequest,
        )

        side = AlpacaSide.BUY if order.side is Side.BUY else AlpacaSide.SELL
        tif = AlpacaTIF(order.time_in_force.value)

        # Bracket / OTO / OCO support — attach stop-loss + take-profit to the
        # parent so Alpaca manages exits server-side. Critical for unattended
        # overnight runs: even if our process dies, the broker still flat-
        # tens at the stop or target.
        order_class = self._map_order_class(order.order_class)
        bracket_kwargs: dict[str, object] = {}
        if order_class is not None:
            from alpaca.trading.requests import StopLossRequest

            if order.stop_loss_price is not None:
                bracket_kwargs["stop_loss"] = StopLossRequest(
                    stop_price=float(order.stop_loss_price)
                )
            if order.take_profit_price is not None:
                bracket_kwargs["take_profit"] = TakeProfitRequest(
                    limit_price=float(order.take_profit_price)
                )
            bracket_kwargs["order_class"] = order_class

        if order.order_type is OrderType.TRAILING_STOP:
            if order.trail_percent is None and order.trail_price is None:
                raise ValueError(
                    "trail_percent or trail_price required for TRAILING_STOP orders"
                )
            req = TrailingStopOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                trail_percent=order.trail_percent,
                trail_price=order.trail_price,
                client_order_id=order.client_order_id,
            )
        elif order.order_type is OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                client_order_id=order.client_order_id,
                **bracket_kwargs,
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
                **bracket_kwargs,
            )

        submitted = self._client.submit_order(order_data=req)
        status = _STATUS_MAP.get(str(submitted.status).lower(), OrderStatus.NEW)
        logger.info(
            "order submitted symbol={} side={} qty={} type={} class={} status={} id={}",
            order.symbol,
            order.side.value,
            order.qty,
            order.order_type.value,
            order.order_class.value,
            status.value,
            submitted.id,
        )
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=str(submitted.id),
            status=status,
        )

    @staticmethod
    def _map_order_class(oc: OrderClass) -> object | None:
        """Translate our OrderClass to Alpaca's enum, or None for SIMPLE."""
        from alpaca.trading.enums import OrderClass as AlpacaOrderClass

        return {
            OrderClass.BRACKET: AlpacaOrderClass.BRACKET,
            OrderClass.OTO: AlpacaOrderClass.OTO,
            OrderClass.OCO: AlpacaOrderClass.OCO,
        }.get(oc)

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
