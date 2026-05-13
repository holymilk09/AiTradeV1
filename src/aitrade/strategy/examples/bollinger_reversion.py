"""Bollinger mean-reversion — long-only.

Edge thesis: liquidity providers paid to fade overreactions. When close
prints below the lower Bollinger band (price > k stddev below SMA),
expect a reversion toward the band's middle within a few bars. Exits to
flat when price reclaims the SMA midline.

Long-only by design: short selling has different mechanics + margin +
shorter borrow constraints, and the engine-first goal is to prove an
edge exists, not to maximize it.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.signal import Direction, Signal


@dataclass
class BollingerReversion:
    symbol: str
    period: int = 20
    num_std: float = 2.0
    strategy_id: str = "bollinger_reversion"
    _closes: deque[float] = field(init=False)
    _last_direction: Direction = field(default=Direction.FLAT, init=False)

    def __post_init__(self) -> None:
        if self.period < 5:
            raise ValueError("period must be >= 5")
        if self.num_std <= 0:
            raise ValueError("num_std must be positive")
        self._closes = deque(maxlen=self.period)

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None
        self._closes.append(bar.close)
        if len(self._closes) < self.period:
            return None

        mid = statistics.fmean(self._closes)
        sd = statistics.pstdev(self._closes)
        if sd == 0:
            return None
        lower = mid - self.num_std * sd
        z = (bar.close - mid) / sd  # standardized distance from mid

        new_dir = self._last_direction
        if self._last_direction is Direction.FLAT and bar.close < lower:
            new_dir = Direction.LONG
        elif self._last_direction is Direction.LONG and bar.close >= mid:
            new_dir = Direction.FLAT

        if new_dir == self._last_direction:
            return None
        self._last_direction = new_dir
        return Signal(
            symbol=self.symbol,
            direction=new_dir,
            strength=min(1.0, abs(z) / (self.num_std * 2)),
            reason=f"close={bar.close:.2f} mid={mid:.2f} z={z:+.2f}",
        )

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
