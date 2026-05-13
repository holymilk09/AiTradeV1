"""Order router: risk gate → broker → trade journal."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from aitrade.brokers.base import BrokerClient
from aitrade.execution.orders import OrderAck, OrderRequest
from aitrade.execution.risk import RiskGate

# After this many consecutive "not tradable" returns for the same symbol,
# stop calling the broker for that symbol until the executor is reset.
# Prevents a paper runner from spamming the journal + the broker once a
# symbol has been firmly classified as non-tradable.
_TRADABILITY_BLACKLIST_AFTER = 3


@dataclass(frozen=True, slots=True)
class SubmitResult:
    accepted: bool
    reason: str = ""
    ack: OrderAck | None = None


class Executor:
    def __init__(self, broker: BrokerClient, risk: RiskGate) -> None:
        self._broker = broker
        self._risk = risk
        self._not_tradable_counts: dict[str, int] = {}
        self._blacklist: set[str] = set()

    @property
    def risk(self) -> RiskGate:
        return self._risk

    def reset_tradability_cache(self) -> None:
        """Clear the per-symbol non-tradable counter + blacklist.

        Call at the start of a new session or when an operator has confirmed
        a halt has lifted. Not called automatically.
        """
        self._not_tradable_counts.clear()
        self._blacklist.clear()

    def submit(
        self,
        order: OrderRequest,
        *,
        reference_price: float,
        current_position_qty: float = 0.0,
    ) -> SubmitResult:
        # Short-circuit: a symbol already classified as non-tradable stays
        # blocked for the rest of the session.
        if order.symbol in self._blacklist:
            reason = (
                f"symbol {order.symbol} blacklisted after "
                f"{_TRADABILITY_BLACKLIST_AFTER} not-tradable returns"
            )
            return SubmitResult(False, reason)

        # Halt / delisting check before risk gate. Cheap (one cached
        # asset lookup) and short-circuits sending to a halted ticker —
        # which Alpaca would reject with a confusing error after the
        # risk-budget tick has already fired.
        try:
            tradable = self._broker.is_tradable(order.symbol)
        except Exception as exc:
            logger.warning(
                "is_tradable raised symbol={} err={}; failing closed",
                order.symbol,
                exc,
            )
            tradable = False
        if not tradable:
            count = self._not_tradable_counts.get(order.symbol, 0) + 1
            self._not_tradable_counts[order.symbol] = count
            reason = f"symbol {order.symbol} not tradable (halted/delisted)"
            if count >= _TRADABILITY_BLACKLIST_AFTER:
                self._blacklist.add(order.symbol)
                logger.warning(
                    "order blocked: {} (blacklisted after {} rejects; "
                    "call executor.reset_tradability_cache() to retry)",
                    reason,
                    count,
                )
            else:
                logger.warning(
                    "order blocked: {} (reject {}/{})",
                    reason,
                    count,
                    _TRADABILITY_BLACKLIST_AFTER,
                )
            return SubmitResult(False, reason)

        # Symbol came back tradable — reset its counter so a transient false
        # negative doesn't accumulate over hours.
        self._not_tradable_counts.pop(order.symbol, None)

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
