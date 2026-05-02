"""Dynamic ATR-based stop / target planning for the floor-trader.

A static "always 1% stop" is wrong in two directions:

* **Too tight in volatile markets** — normal noise stops us out before the
  thesis can play out.
* **Too wide in calm markets** — we give back disproportionate gains on
  positions whose normal range is half a percent.

The right primitive is **ATR multiples**: a 1.5× ATR stop is *the same
amount of normal-noise tolerance* whether ATR is $0.20 or $5.00.

This module produces a :class:`StopPlan` from an entry price + the
candidate's daily ATR + a regime-aware multiplier. The floor-trader
suggests ``stop_atr_mult`` (typically 1.0-2.5) and ``target_atr_mult``
(typically 2.0-4.0); the engine resolves them into absolute prices here,
applies a hard reward/risk floor, and emits a :class:`StopPlan` that's
threaded into the bracket order.

Why a separate module:

* Keeps ATR math + regime adjustments out of the engine_runner.
* Single test surface — ``compute_stop_plan`` is pure.
* Future variants (chandelier exit, parabolic SAR, structure-based stops)
  slot in here without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.execution.orders import Side


@dataclass(frozen=True, slots=True)
class StopPlan:
    """Resolved stop-loss + take-profit prices for a single bracket order.

    The plan's ``trail_percent`` is non-zero when the engine should later
    convert the static stop into a trailing stop (typically once the trade
    is profitable by ≥1× ATR). Zero means "static stop only".
    """

    stop_loss_price: float
    take_profit_price: float
    trail_percent: float = 0.0
    rationale: str = ""

    def reward_risk_ratio(self, entry: float, side: Side) -> float:
        """Compute the resolved reward:risk ratio, useful for journaling."""
        if side is Side.BUY:
            risk = max(0.0, entry - self.stop_loss_price)
            reward = max(0.0, self.take_profit_price - entry)
        else:  # SELL / short
            risk = max(0.0, self.stop_loss_price - entry)
            reward = max(0.0, entry - self.take_profit_price)
        if risk == 0.0:
            return 0.0
        return reward / risk


# Minimum acceptable reward-to-risk. We refuse plans with R:R < this; better
# to pass than to take a structurally bad trade.
_MIN_REWARD_RISK = 1.5

# Minimum stop distance as a fraction of price — guards against degenerate
# ATR readings on illiquid names where ATR can collapse to a few cents.
_MIN_STOP_PCT = 0.005  # 0.5%

# When VIX is elevated, widen stops to avoid noise-fueled stop-outs. Multiplier
# is a smooth ramp on the VIX value: at VIX=15 we use 1.0x, at VIX=30 1.4x.
_VIX_BASE = 15.0
_VIX_SLOPE = 0.027  # (1.4 - 1.0) / (30 - 15)


def _vix_widening_factor(vix: float) -> float:
    """Multiplier on stop *distance* given the current VIX read.

    Returns 1.0 at calm (VIX ≤ 15), scales linearly to ~1.4x at VIX 30,
    then clamps. Targets widen by the *same* factor so reward:risk
    is preserved — the trade just sits inside a bigger, slower frame.
    """
    if vix <= _VIX_BASE:
        return 1.0
    factor = 1.0 + _VIX_SLOPE * (vix - _VIX_BASE)
    return min(factor, 1.4)


def compute_stop_plan(
    *,
    entry_price: float,
    atr: float,
    side: Side,
    stop_atr_mult: float,
    target_atr_mult: float,
    vix: float | None = None,
    enable_trailing: bool = True,
) -> StopPlan | None:
    """Resolve ATR multiples into absolute stop / target prices.

    Returns ``None`` when the resulting plan fails the minimum-reward-risk
    floor — the engine treats that as "skip this trade", same as a stale-
    data abort. Better to take fewer, structurally sound trades than to
    fill a journal with mathematically bad ones.

    Parameters
    ----------
    entry_price:
        The reference price the engine will quote against. Must be positive.
    atr:
        Average True Range from the daily timeframe — the volatility scale.
    side:
        BUY (long) or SELL (short). Determines which side the stop sits on.
    stop_atr_mult:
        How many ATRs below (long) / above (short) entry to place the stop.
        Floor-trader default is 1.5; tighter for high-conviction entries,
        wider for choppy names.
    target_atr_mult:
        How many ATRs the take-profit sits *favorable* of entry.
    vix:
        Current VIX (or proxy). When ≥15, stops widen via
        ``_vix_widening_factor`` so we don't get stopped on regime noise.
        Pass ``None`` to disable the adjustment (e.g. in tests).
    enable_trailing:
        When True, the plan suggests a trailing-stop conversion at the
        per-side ATR distance once the trade is profitable. Engine wiring
        decides when to actually flip from static → trailing.
    """
    if entry_price <= 0:
        return None
    if atr <= 0:
        return None
    if stop_atr_mult <= 0 or target_atr_mult <= 0:
        return None

    width = _vix_widening_factor(vix) if vix is not None else 1.0
    raw_stop_dist = atr * stop_atr_mult * width
    raw_target_dist = atr * target_atr_mult * width

    # Apply a minimum stop distance — protects against degenerate ATR on
    # quiet or thinly-traded names where ATR collapses near zero.
    min_dist = entry_price * _MIN_STOP_PCT
    stop_dist = max(raw_stop_dist, min_dist)
    target_dist = max(raw_target_dist, min_dist * (target_atr_mult / stop_atr_mult))

    if side is Side.BUY:
        stop_price = entry_price - stop_dist
        target_price = entry_price + target_dist
    else:  # SELL / short
        stop_price = entry_price + stop_dist
        target_price = entry_price - target_dist

    # Sanity: prices must be positive (catches absurd ATR on cheap names).
    if stop_price <= 0 or target_price <= 0:
        return None

    plan = StopPlan(
        stop_loss_price=round(stop_price, 4),
        take_profit_price=round(target_price, 4),
        trail_percent=0.0,
        rationale=(
            f"stop={stop_atr_mult:.2f}xATR target={target_atr_mult:.2f}xATR "
            f"vix_factor={width:.2f}"
        ),
    )

    # Reject mathematically bad trades — better to pass than fill the
    # journal with structurally losing setups.
    rr = plan.reward_risk_ratio(entry_price, side)
    if rr < _MIN_REWARD_RISK:
        return None

    if enable_trailing:
        # Trailing distance is the same ATR multiple, expressed as a percent
        # of entry price so Alpaca's trail_percent contract maps cleanly.
        trail_pct = (atr * stop_atr_mult * width / entry_price) * 100.0
        plan = StopPlan(
            stop_loss_price=plan.stop_loss_price,
            take_profit_price=plan.take_profit_price,
            trail_percent=round(trail_pct, 3),
            rationale=plan.rationale,
        )

    return plan


__all__ = [
    "StopPlan",
    "compute_stop_plan",
]
