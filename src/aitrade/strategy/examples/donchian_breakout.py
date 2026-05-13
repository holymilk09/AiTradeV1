"""Donchian-channel breakout — long-only.

Edge thesis: a new N-bar high marks a regime where prior resistance has
cleared and trend-followers join. Exit when price closes below an
exit-channel low (Donchian exit), the classic turtle pattern.

Asymmetric entry/exit windows reduce whipsaw: enter on a longer-lookback
break (commitment), exit on a shorter-lookback break (responsiveness).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.signal import Direction, Signal


@dataclass
class DonchianBreakout:
    symbol: str
    entry_period: int = 20
    exit_period: int = 10
    strategy_id: str = "donchian_breakout"
    _entry_highs: deque[float] = field(init=False)
    _exit_lows: deque[float] = field(init=False)
    _last_direction: Direction = field(default=Direction.FLAT, init=False)

    def __post_init__(self) -> None:
        if self.entry_period < 2 or self.exit_period < 2:
            raise ValueError("periods must be >= 2")
        if self.exit_period >= self.entry_period:
            raise ValueError("exit_period must be < entry_period")
        # Track previous-bar windows so the current bar's close is compared
        # against a true N-bar prior high/low (no lookahead).
        self._entry_highs = deque(maxlen=self.entry_period)
        self._exit_lows = deque(maxlen=self.exit_period)

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None

        new_dir = self._last_direction
        if len(self._entry_highs) >= self.entry_period:
            prior_high = max(self._entry_highs)
            prior_low = min(self._exit_lows) if self._exit_lows else bar.close
            if self._last_direction is Direction.FLAT and bar.close > prior_high:
                new_dir = Direction.LONG
            elif self._last_direction is Direction.LONG and bar.close < prior_low:
                new_dir = Direction.FLAT

        # Slide windows AFTER the decision so we compare current close to a
        # strictly-prior window.
        self._entry_highs.append(bar.high)
        self._exit_lows.append(bar.low)

        if new_dir == self._last_direction:
            return None
        self._last_direction = new_dir
        return Signal(
            symbol=self.symbol,
            direction=new_dir,
            strength=0.7,
            reason=f"close={bar.close:.2f} donchian_break",
        )

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
