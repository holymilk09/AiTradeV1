"""Z-score volume anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from aitrade.data.models import Bar
from aitrade.patterns.base import PatternSignal, clamp01
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class UnusualVolumeDetector:
    """Fires when the latest bar's volume is a statistical outlier.

    Direction is LONG if the bar closed up vs the prior close, otherwise
    FLAT — heavy selling is information, but it isn't a long entry.
    """

    name: str = "unusual_volume"
    window: int = 20
    z_threshold: float = 2.0

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a signal if the latest bar's volume z-score exceeds the threshold."""
        if self.window < 2:
            return None
        if len(bars) < self.window + 1:
            return None

        latest = bars[-1]
        prior = bars[-self.window - 1 : -1]
        vols = [b.volume for b in prior]
        mean = sum(vols) / len(vols)
        var = sum((v - mean) ** 2 for v in vols) / len(vols)
        std = sqrt(var)
        if std <= 0:
            return None
        z = (latest.volume - mean) / std
        if z <= self.z_threshold:
            return None

        prev_close = prior[-1].close
        direction = Direction.LONG if latest.close > prev_close else Direction.FLAT
        score = clamp01(z / 5.0)

        return PatternSignal(
            name=self.name,
            symbol=latest.symbol,
            score=score,
            direction=direction,
            timestamp=latest.timestamp,
            evidence={
                "z_score": z,
                "vol_mean": mean,
                "vol_std": std,
            },
        )
