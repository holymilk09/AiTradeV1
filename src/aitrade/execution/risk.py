"""Pre-trade risk gates. Every order passes through ``check(...)`` before submit.

Policy:
  - per-order notional cap
  - per-symbol gross exposure cap
  - daily realized PnL floor (kill switch when crossed)
  - order rate limit (tokens-per-minute)
  - optional file-based kill switch: touching ``kill_switch_path`` halts trading
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from aitrade.execution.orders import OrderRequest, Side


class RiskViolation(Exception):
    """Raised (or wrapped in a Decision) when an order fails a gate."""


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    reason: str = ""


@dataclass
class RiskGate:
    max_position_usd: float
    max_daily_loss_usd: float
    max_orders_per_min: int
    kill_switch_path: Path | None = None

    _order_times: deque[float] = field(default_factory=deque, init=False)
    _daily_realized_pnl: float = field(default=0.0, init=False)
    _halted: bool = field(default=False, init=False)

    def record_realized_pnl(self, pnl: float) -> None:
        self._daily_realized_pnl += pnl
        if self._daily_realized_pnl <= -abs(self.max_daily_loss_usd):
            self._halted = True

    def reset_day(self) -> None:
        self._daily_realized_pnl = 0.0
        self._halted = False

    @property
    def halted(self) -> bool:
        return self._halted

    def _kill_switch_active(self) -> bool:
        return self.kill_switch_path is not None and self.kill_switch_path.exists()

    def _rate_limited(self) -> bool:
        now = monotonic()
        cutoff = now - 60.0
        while self._order_times and self._order_times[0] < cutoff:
            self._order_times.popleft()
        return len(self._order_times) >= self.max_orders_per_min

    def check(
        self,
        order: OrderRequest,
        *,
        reference_price: float,
        current_position_qty: float = 0.0,
    ) -> Decision:
        if self._halted:
            return Decision(False, "halted: daily loss floor breached")
        if self._kill_switch_active():
            return Decision(False, f"kill switch file present: {self.kill_switch_path}")
        if self._rate_limited():
            return Decision(False, f"rate limit: >{self.max_orders_per_min}/min")
        if reference_price <= 0:
            return Decision(False, "non-positive reference price")

        signed_qty = order.qty if order.side is Side.BUY else -order.qty
        projected_qty = current_position_qty + signed_qty
        projected_notional = abs(projected_qty) * reference_price
        if projected_notional > self.max_position_usd:
            return Decision(
                False,
                f"projected notional ${projected_notional:,.2f} exceeds cap "
                f"${self.max_position_usd:,.2f}",
            )
        return Decision(True)

    def commit(self) -> None:
        """Record that an order was sent. Call only after a passing ``check``."""
        self._order_times.append(monotonic())
