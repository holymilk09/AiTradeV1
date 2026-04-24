"""N-day high breakout with volume confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.data.models import Bar
from aitrade.patterns.base import PatternSignal, clamp01
from aitrade.strategy.indicators import sma
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class BreakoutDetector:
    """Fires when the latest close pierces the prior ``lookback``-bar high.

    Requires a volume kicker: the breakout bar's volume must clear the 20-day
    SMA of volume by ``vol_mult``. Without that, breakouts are noise.
    """

    name: str = "breakout"
    lookback: int = 20
    vol_mult: float = 1.5

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a LONG signal on a confirmed N-day high breakout."""
        if self.lookback < 2:
            return None
        # Need ``lookback`` prior closes + the breakout bar; also 20 prior
        # bars for the volume baseline (which here is the same window since
        # lookback defaults to 20).
        min_required = max(self.lookback, 20) + 1
        if len(bars) < min_required:
            return None

        latest = bars[-1]
        prior_closes = [b.close for b in bars[-self.lookback - 1 : -1]]
        prior_high = max(prior_closes)
        if latest.close <= prior_high:
            return None

        vol_window = [b.volume for b in bars[-21:-1]]
        avg_vol = sma(vol_window, 20)
        if avg_vol is None or avg_vol <= 0:
            return None
        vol_ratio = latest.volume / avg_vol
        if vol_ratio <= self.vol_mult:
            return None

        breakout_pct = (latest.close - prior_high) / prior_high if prior_high else 0.0
        score = clamp01(breakout_pct * 10.0)

        return PatternSignal(
            name=self.name,
            symbol=latest.symbol,
            score=score,
            direction=Direction.LONG,
            timestamp=latest.timestamp,
            evidence={
                "prior_high": prior_high,
                "breakout_pct": breakout_pct,
                "vol_ratio": vol_ratio,
            },
        )
