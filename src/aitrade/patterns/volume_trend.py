"""Multi-day price-up + volume-expansion detector."""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.data.models import Bar
from aitrade.patterns.base import PatternSignal, clamp01
from aitrade.strategy.indicators import sma
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class VolumePriceTrendDetector:
    """Fires when the last ``days`` bars all close up on above-average volume.

    A quick-and-dirty institutional-accumulation tell: ``days`` consecutive
    higher closes paired with volume above the 20-day SMA times ``vol_mult``.
    """

    name: str = "volume_price_trend"
    days: int = 3
    vol_mult: float = 1.15

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a LONG signal when every recent bar is up on heavy volume."""
        if self.days < 1:
            return None
        # Need 20 bars for the volume baseline + ``days`` recent bars + 1 prior
        # close for the first "up" comparison.
        min_required = 20 + self.days + 1
        if len(bars) < min_required:
            return None

        recent = bars[-self.days :]
        prior_closes = [bars[-self.days - 1 + i].close for i in range(self.days)]

        # Every recent bar's close must exceed the bar before it.
        for bar, prev_close in zip(recent, prior_closes, strict=True):
            if bar.close <= prev_close:
                return None

        vol_ratios: list[float] = []
        for i, bar in enumerate(recent):
            # Volume baseline excludes the bar itself: prior 20 bars only.
            end_idx = len(bars) - self.days + i
            window = [b.volume for b in bars[end_idx - 20 : end_idx]]
            avg_vol = sma(window, 20)
            if avg_vol is None or avg_vol <= 0:
                return None
            ratio = bar.volume / avg_vol
            if ratio <= self.vol_mult:
                return None
            vol_ratios.append(ratio)

        avg_vol_ratio = sum(vol_ratios) / len(vol_ratios)
        first_close = prior_closes[0]
        last_close = recent[-1].close
        pct_change = (last_close - first_close) / first_close if first_close else 0.0
        score = clamp01(avg_vol_ratio / 2.0)

        return PatternSignal(
            name=self.name,
            symbol=recent[-1].symbol,
            score=score,
            direction=Direction.LONG,
            timestamp=recent[-1].timestamp,
            evidence={
                "days_up": float(self.days),
                "avg_vol_ratio": avg_vol_ratio,
                "pct_change": pct_change,
            },
        )
