"""ml_signal: lightweight feature-based score with a fitted linear weight vector.

V1 is intentionally minimal — a logistic-style score over four
hand-engineered numeric features, with weights either set to a sensible
default or fitted offline against forward returns via
``aitrade.signals.calibration``. No scikit-learn dependency; the upgrade
path to GBM is straightforward when training data justifies it.

Features (computed from the recent bar window):
- ret_5: 5-bar % return
- ret_20: 20-bar % return
- vol_20_z: stdev of 20-bar returns standardized by 60-bar history (regime)
- range_20_z: avg high-low range over 20 bars standardized similarly
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent

_FEATURES = ("ret_5", "ret_20", "vol_20_z", "range_20_z")
_DEFAULT_COEFS: dict[str, float] = {
    "ret_5": 1.2,
    "ret_20": 0.6,
    "vol_20_z": -0.4,
    "range_20_z": 0.2,
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _compute_features(
    closes: list[float], highs: list[float], lows: list[float]
) -> dict[str, float]:
    if len(closes) < 60:
        return dict.fromkeys(_FEATURES, 0.0)

    ret_5 = (closes[-1] / closes[-6] - 1.0) if closes[-6] > 0 else 0.0
    ret_20 = (closes[-1] / closes[-21] - 1.0) if closes[-21] > 0 else 0.0

    returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    recent_vol = statistics.pstdev(returns[-20:]) if len(returns) >= 20 else 0.0
    hist_returns = returns[-60:]
    hist_vol_mean = statistics.fmean(
        statistics.pstdev(hist_returns[i - 20 : i]) for i in range(20, len(hist_returns) + 1)
    )
    hist_vol_sd = statistics.pstdev(
        [
            statistics.pstdev(hist_returns[i - 20 : i])
            for i in range(20, len(hist_returns) + 1)
        ]
    ) or 1e-9
    vol_20_z = (recent_vol - hist_vol_mean) / hist_vol_sd

    ranges = [h - low for h, low in zip(highs, lows, strict=True)]
    recent_range = statistics.fmean(ranges[-20:])
    hist_ranges = ranges[-60:]
    hist_range_mean = statistics.fmean(hist_ranges)
    hist_range_sd = statistics.pstdev(hist_ranges) or 1e-9
    range_20_z = (recent_range - hist_range_mean) / hist_range_sd

    return {
        "ret_5": ret_5,
        "ret_20": ret_20,
        "vol_20_z": vol_20_z,
        "range_20_z": range_20_z,
    }


@dataclass
class MlSignalScorer:
    name: str = "ml_signal"
    default_weight: float = 0.25
    coefficients: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_COEFS)
    )
    bias: float = 0.0

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        if len(ctx.bars) < 60:
            return ConvictionComponent(
                score=0.0,
                weight=self.default_weight,
                evidence="insufficient history (<60 bars)",
            )
        closes = [b.close for b in ctx.bars]
        highs = [b.high for b in ctx.bars]
        lows = [b.low for b in ctx.bars]
        features = _compute_features(closes, highs, lows)
        logit = self.bias + sum(
            features[k] * self.coefficients.get(k, 0.0) for k in _FEATURES
        )
        p = _sigmoid(logit)
        return ConvictionComponent(
            score=p,
            weight=self.default_weight,
            evidence=(
                f"ret5={features['ret_5']:+.3f} ret20={features['ret_20']:+.3f} "
                f"volZ={features['vol_20_z']:+.2f} rngZ={features['range_20_z']:+.2f}"
            ),
            features=features,
        )
