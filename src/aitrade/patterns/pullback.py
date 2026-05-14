"""Pullback-to-50-SMA-in-uptrend detector."""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.data.models import Bar
from aitrade.patterns.base import PatternSignal, clamp01
from aitrade.strategy.indicators import rsi, sma
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class PullbackDetector:
    """Fires on a shallow pullback to the 50-SMA inside a confirmed uptrend.

    Uptrend: 50-SMA above 200-SMA. Pullback: latest close within ``pct``
    percent of the 50-SMA. Confirmation: RSI(14) cooled into ``[rsi_lo, rsi_hi]``
    — oversold-ish but not crashing.
    """

    name: str = "pullback"
    pct: float = 1.5
    rsi_lo: float = 30.0
    rsi_hi: float = 45.0

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a LONG signal when an uptrend pulls back to its 50-SMA."""
        if len(bars) < 200:
            return None
        if self.rsi_hi <= self.rsi_lo:
            return None

        closes = [b.close for b in bars]
        sma50 = sma(closes, 50)
        sma200 = sma(closes, 200)
        rsi14 = rsi(closes, 14)
        if sma50 is None or sma200 is None or rsi14 is None:
            return None
        if sma50 <= sma200:
            return None

        latest = bars[-1]
        if sma50 == 0:
            return None
        dist_pct = abs(latest.close - sma50) / sma50 * 100.0
        if dist_pct > self.pct:
            return None
        if not (self.rsi_lo <= rsi14 <= self.rsi_hi):
            return None

        score = clamp01((self.rsi_hi - rsi14) / (self.rsi_hi - self.rsi_lo))

        return PatternSignal(
            name=self.name,
            symbol=latest.symbol,
            score=score,
            direction=Direction.LONG,
            timestamp=latest.timestamp,
            evidence={
                "rsi_14": rsi14,
                "dist_to_sma50_pct": dist_pct,
                "sma50": sma50,
                "sma200": sma200,
            },
        )
