"""SMA crossover — reference strategy to exercise the full loop.

Long when fast SMA > slow SMA; flat otherwise. Not a real edge — a wiring test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.signal import Direction, Signal


@dataclass
class SmaCrossover:
    symbol: str
    fast: int = 10
    slow: int = 30
    strategy_id: str = "sma_crossover"
    _fast_window: deque[float] = field(init=False)
    _slow_window: deque[float] = field(init=False)
    _last_direction: Direction = field(default=Direction.FLAT, init=False)

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast must be < slow")
        self._fast_window = deque(maxlen=self.fast)
        self._slow_window = deque(maxlen=self.slow)

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None
        self._fast_window.append(bar.close)
        self._slow_window.append(bar.close)
        if len(self._slow_window) < self.slow:
            return None

        fast_avg = sum(self._fast_window) / len(self._fast_window)
        slow_avg = sum(self._slow_window) / len(self._slow_window)
        new_dir = Direction.LONG if fast_avg > slow_avg else Direction.FLAT
        if new_dir == self._last_direction:
            return None
        self._last_direction = new_dir
        return Signal(
            symbol=self.symbol,
            direction=new_dir,
            strength=min(1.0, abs(fast_avg - slow_avg) / slow_avg),
            reason=f"fast={fast_avg:.2f} slow={slow_avg:.2f}",
        )

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
