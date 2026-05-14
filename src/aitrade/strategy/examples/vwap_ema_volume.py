"""VWAP reclaim with EMA stack and volume confirmation — long-only.

Edge thesis. Three of the most-watched intraday tools by professional
desks, applied here on daily bars:

  - **VWAP** — volume-weighted typical price. "Fair value" anchor.
    Crossing it from below with volume = institutional buying. Stocks
    above their rolling VWAP have favorable accumulation; below = the
    opposite. Most-watched single intraday level on Wall Street.
  - **EMA stack (8/21)** — short-term trend filter. The 8-day EMA over
    the 21-day EMA is the canonical "uptrend in progress" signal that
    swing desks use to fade strength or buy dips.
  - **Volume confirmation** — current volume > 1.3 × 20-day average is
    the conventional threshold for "real" participation, not noise.

Combining all three forces a setup to satisfy *price-action*, *trend*,
AND *flow* before firing. That tight conjunction is the point — each
filter alone is noisy; together they're meaningfully selective.

Entry (FLAT → LONG):
  1. close > rolling 20-day VWAP   (above fair value)
  2. previous close was at-or-below VWAP, current close is above
     (the *reclaim* — a transition, not just a state)
  3. 8-EMA > 21-EMA                (short-term uptrend in place)
  4. volume_ratio > 1.3            (above-average participation)

Exit (LONG → FLAT):
  - close drops back below 8-EMA, OR
  - 8-EMA crosses below 21-EMA

Long-only by design — Phase 1 doesn't model shorts, and a short-side
version of this trade (reject from VWAP + EMA stack bearish + volume)
would be a separate strategy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.indicators import (
    ema,
    rolling_vwap,
    volume_ratio,
)
from aitrade.strategy.signal import Direction, Signal


@dataclass
class VwapEmaVolume:
    symbol: str
    vwap_window: int = 20
    ema_fast: int = 8
    ema_slow: int = 21
    vol_avg_window: int = 20
    min_volume_ratio: float = 1.3
    strategy_id: str = "vwap_ema_volume"
    _bars: deque[Bar] = field(init=False)
    _last_direction: Direction = field(default=Direction.FLAT, init=False)

    def __post_init__(self) -> None:
        if self.vwap_window < 5:
            raise ValueError("vwap_window must be >= 5")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be < ema_slow")
        if self.min_volume_ratio <= 0:
            raise ValueError("min_volume_ratio must be positive")
        # Keep enough history for the slowest indicator (slow EMA needs ~3×
        # its window to stabilize) plus the VWAP/volume windows.
        cap = max(self.vwap_window, self.vol_avg_window, self.ema_slow * 3, 60)
        self._bars = deque(maxlen=cap)

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None
        self._bars.append(bar)
        bars = list(self._bars)
        if len(bars) < max(self.vwap_window, self.vol_avg_window, self.ema_slow) + 2:
            return None

        closes = [b.close for b in bars]
        prev_bars = bars[:-1]
        prev_closes = closes[:-1]
        curr_vwap = rolling_vwap(bars, self.vwap_window)
        prev_vwap = rolling_vwap(prev_bars, self.vwap_window)
        fast = ema(closes, self.ema_fast)
        slow = ema(closes, self.ema_slow)
        prev_fast = ema(prev_closes, self.ema_fast)
        prev_slow = ema(prev_closes, self.ema_slow)
        vol_r = volume_ratio(bars, self.vol_avg_window)
        if None in (curr_vwap, prev_vwap, fast, slow, prev_fast, prev_slow, vol_r):
            return None
        # Narrow types — each was None-checked above.
        assert curr_vwap is not None
        assert prev_vwap is not None
        assert fast is not None
        assert slow is not None
        assert prev_fast is not None
        assert prev_slow is not None
        assert vol_r is not None

        curr_close = bar.close
        prev_close = prev_bars[-1].close

        # --- Entry rule ---
        if self._last_direction is Direction.FLAT:
            reclaim = prev_close <= prev_vwap and curr_close > curr_vwap
            ema_up = fast > slow
            flow_ok = vol_r >= self.min_volume_ratio
            if reclaim and ema_up and flow_ok:
                self._last_direction = Direction.LONG
                return Signal(
                    symbol=self.symbol,
                    direction=Direction.LONG,
                    strength=min(1.0, (vol_r - 1.0) / 2.0 + 0.3),
                    reason=(
                        f"vwap_reclaim close={curr_close:.2f} vwap={curr_vwap:.2f} "
                        f"ema8={fast:.2f}>ema21={slow:.2f} vol_ratio={vol_r:.2f}"
                    ),
                )
            return None

        # --- Exit rule (already long) ---
        ema_cross_down = prev_fast >= prev_slow and fast < slow
        close_below_fast_ema = curr_close < fast
        if close_below_fast_ema or ema_cross_down:
            self._last_direction = Direction.FLAT
            reason = "ema8_cross_down" if ema_cross_down else "close<ema8"
            return Signal(
                symbol=self.symbol,
                direction=Direction.FLAT,
                strength=0.5,
                reason=f"{reason} close={curr_close:.2f} ema8={fast:.2f} ema21={slow:.2f}",
            )
        return None

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
