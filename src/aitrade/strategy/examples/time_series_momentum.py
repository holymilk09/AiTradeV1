"""Time-series momentum (12-1) — long-only.

Edge thesis. The most-cited single-asset factor in academic finance:
the sign of a stock's trailing (lookback - skip) return predicts the
sign of its next-month return. Documented across asset classes back
to 1903 (Hurst-Ooi-Pedersen 2017; Moskowitz-Ooi-Pedersen 2012;
Asness-Moskowitz-Pedersen 2013).

Classical formulation: take the trailing 12-month return, but SKIP the
most recent 1 month to avoid being contaminated by the well-known
1-month reversal effect (Jegadeesh 1990). So the signal at day t is

    sgn( close[t - skip] / close[t - lookback] - 1 )

Long when positive, flat when negative. **Long-only** in this
implementation — Phase 1 doesn't model shorts.

This is the canonical "trend following" trade implemented on a single
name. It's mechanically orthogonal to bollinger_reversion: bollinger
profits from short-term mean-reversion against the medium-term trend,
this profits from medium-term trend persistence. If both edges are
real, their P&Ls should be lightly correlated and they're worth
running together.

Why no rebalance gating. The Strategy Protocol gets one bar at a time;
we emit a direction-change Signal whenever the trailing-return sign
flips. The runner's executor handles the actual order placement at
next-bar open, so position turnover is naturally bounded by how often
the sign flips — typically a few times per year on liquid names.

Param defaults (lookback=252, skip=21) match the standard 12-1 month
specification on daily bars. They are intentionally not tunable per
backtest: the academic robustness comes from those specific windows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.signal import Direction, Signal


@dataclass
class TimeSeriesMomentum:
    symbol: str
    lookback: int = 252
    skip: int = 21
    strategy_id: str = "time_series_momentum"
    _closes: deque[float] = field(init=False)
    _last_direction: Direction = field(default=Direction.FLAT, init=False)

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be >= 5 (need enough history)")
        if self.skip < 1:
            raise ValueError("skip must be >= 1")
        if self.skip >= self.lookback:
            raise ValueError("skip must be < lookback")
        # Hold one extra slot so we can index at -lookback exactly.
        self._closes = deque(maxlen=self.lookback + 1)

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None
        self._closes.append(bar.close)
        # Need enough history to compute close[t-skip] / close[t-lookback].
        if len(self._closes) < self.lookback + 1:
            return None

        # closes is indexed 0..len-1 with the most-recent at len-1.
        # close[t-skip] = closes[-(skip+1)]  (e.g. skip=21 → closes[-22])
        # close[t-lookback] = closes[0]      (oldest in the deque)
        close_skip = self._closes[-(self.skip + 1)]
        close_lookback = self._closes[0]
        trailing_ret = (close_skip / close_lookback) - 1.0

        new_dir = Direction.LONG if trailing_ret > 0 else Direction.FLAT
        if new_dir == self._last_direction:
            return None
        self._last_direction = new_dir

        return Signal(
            symbol=self.symbol,
            direction=new_dir,
            # Strength scaled by magnitude of trailing return — capped at 1.0
            # to keep it in the [0, 1] contract the conviction component
            # composite assumes elsewhere in the codebase.
            strength=min(1.0, abs(trailing_ret) * 2.0),
            reason=f"trailing_ret_{self.lookback}_{self.skip}={trailing_ret:+.2%}",
        )

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
