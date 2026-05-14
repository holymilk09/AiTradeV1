"""Market-wide regime classification.

A coarse 2x2 split over (direction, volatility):
- direction: SPY day-over-day change >= 0 → risk-on, else risk-off.
- volatility: VIX < 20 → low-vol, else high-vol.

These four buckets are an opinionated but standard heuristic used to gate
strategy aggressiveness. The classifier itself is a pure function so it can
be unit-tested without touching any data feed.
"""

from __future__ import annotations

from enum import Enum


class Regime(str, Enum):
    """Coarse market regime label.

    Combines direction (risk-on/off) with volatility (low/high) into four
    mutually-exclusive buckets. Strategies use this to size positions or
    suppress entries during hostile conditions.
    """

    RISK_ON_LOW_VOL = "risk_on_low_vol"
    RISK_ON_HIGH_VOL = "risk_on_high_vol"
    RISK_OFF_LOW_VOL = "risk_off_low_vol"
    RISK_OFF_HIGH_VOL = "risk_off_high_vol"


def classify_regime(spy_change_pct: float, vix: float) -> Regime:
    """Bucket the market into a :class:`Regime`.

    Parameters
    ----------
    spy_change_pct:
        SPY's percentage change vs. the prior trading day's close. Positive
        (or zero) means risk-on; negative means risk-off.
    vix:
        Latest VIX (or proxy) value. Below 20 is low-vol; 20 or above is
        high-vol.
    """

    risk_on = spy_change_pct >= 0
    low_vol = vix < 20
    if risk_on and low_vol:
        return Regime.RISK_ON_LOW_VOL
    if risk_on and not low_vol:
        return Regime.RISK_ON_HIGH_VOL
    if not risk_on and low_vol:
        return Regime.RISK_OFF_LOW_VOL
    return Regime.RISK_OFF_HIGH_VOL
