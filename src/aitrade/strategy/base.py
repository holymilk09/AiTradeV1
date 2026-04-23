"""Strategy Protocol. Implementations drive signals from market data."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.signal import Signal


@runtime_checkable
class Strategy(Protocol):
    """Strategy lifecycle hooks. Called by the runner loop.

    Hooks may return a Signal or None. None means no-op.
    """

    strategy_id: str

    def on_bar(self, bar: Bar) -> Signal | None: ...

    def on_quote(self, quote: Quote) -> Signal | None: ...

    def on_fill(self, fill: Fill) -> None: ...
