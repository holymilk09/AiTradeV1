"""vix_regime: volatility-regime conviction premium.

Empirical finding (2026-05-14 research, see docs/RESEARCH.md EXP-004):
on every tested liquid US index and large-cap (ex NVDA), the overnight
session's risk-adjusted return is dramatically *regime-dependent*:

  - Low-VIX regime (VIXY bottom quintile, 252-day rolling): net Sharpe
    of overnight-long is +2.1 (IWM) to +4.4 (QQQ); single-name AAPL/TSLA
    hit +4.0 / +3.2 net.
  - High-VIX regime (top decile): net Sharpe is consistently -1.8 to
    -4.3. The overnight drift *reverses*. Don't hold overnight.

NVDA inverts the pattern — likely because intraday-momentum dynamics
on the 2024-2026 AI-rally window swamped the textbook overnight bias.
The scorer treats NVDA the same as anything else; the LLM reasoner is
expected to pattern-match exceptions from `recent_fills_summary`.

This scorer surfaces the regime as a conviction-component input the
LLM reasoner (or the composite ranker) can read. It does NOT
unilaterally enter trades — that's the strategy/reasoner layer's job.

Volatility-regime proxy is the symbol's own 20-day realized vol
percentile against its 252-day rolling distribution. We use *realized
vol* rather than the VIXY proxy directly so the scorer is
self-contained — no extra bar fetch required. Realized vol on liquid
US equities tracks VIX with correlation ~0.7-0.9 during stress.
"""

from __future__ import annotations

import math

from aitrade.signals.components.base import ScoringContext
from aitrade.signals.conviction import ConvictionComponent

# Empirical thresholds from EXP-004. Percentile of the 20d realized
# vol against the 252d trailing distribution.
LOW_VOL_PCT = 20.0
HIGH_VOL_PCT = 80.0

# Need at least this many bars to compute a meaningful percentile.
MIN_BARS_FOR_PCT = 60
RVOL_WINDOW = 20


def _realized_vol(closes: list[float]) -> float | None:
    """Annualized stdev of daily log returns over the window."""
    if len(closes) < 2:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    n = len(rets)
    if n < 2:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def _rolling_percentile(window: list[float], value: float) -> float:
    """Percentile of `value` in `window` (0–100). Empty window → 50."""
    if not window:
        return 50.0
    below = sum(1 for w in window if w < value)
    return (below / len(window)) * 100.0


class VixRegimeScorer:
    """Score a setup by its volatility-regime favorability.

    score = 1.0 when realized vol is in the bottom quintile (LOW_VOL_PCT
              of the 252d distribution) — the regime where overnight bias
              has positive expectancy.
    score = 0.0 when realized vol is in the top quintile (≥HIGH_VOL_PCT) —
              regime where overnight bias inverts; the composite should
              be penalized.
    score = linear interpolation between, so mid-regime gets a moderate
              premium without forcing a binary cliff.

    Features exposed:
      - rvol_20d        : annualized 20-day realized vol (e.g. 0.18 = 18%)
      - rvol_pct        : 252d-rolling percentile (0–100)
      - regime          : "low" | "mid" | "high"
    """

    name: str = "vix_regime"
    default_weight: float = 0.2

    def score(self, ctx: ScoringContext) -> ConvictionComponent:
        bars = ctx.bars
        if len(bars) < MIN_BARS_FOR_PCT:
            return ConvictionComponent(
                score=0.5,
                weight=self.default_weight,
                evidence=f"insufficient history ({len(bars)} bars)",
                features={"rvol_20d": 0.0, "rvol_pct": 50.0},
            )

        closes = [b.close for b in bars]

        # Current realized vol (last RVOL_WINDOW bars).
        current_rvol = _realized_vol(closes[-(RVOL_WINDOW + 1):])
        if current_rvol is None or current_rvol == 0:
            return ConvictionComponent(
                score=0.5,
                weight=self.default_weight,
                evidence="rvol undefined (zero-variance window)",
                features={"rvol_20d": 0.0, "rvol_pct": 50.0},
            )

        # Build the 252-day distribution of trailing realized vols.
        distribution: list[float] = []
        max_lookback = min(len(closes), 252 + RVOL_WINDOW)
        for end in range(RVOL_WINDOW + 1, max_lookback + 1):
            window_closes = closes[-end:][: RVOL_WINDOW + 1]
            rv = _realized_vol(window_closes)
            if rv is not None and rv > 0:
                distribution.append(rv)

        pct = _rolling_percentile(distribution, current_rvol)

        # Linear scale: pct=LOW_VOL_PCT → score 1.0; pct=HIGH_VOL_PCT → 0.0.
        if pct <= LOW_VOL_PCT:
            score = 1.0
        elif pct >= HIGH_VOL_PCT:
            score = 0.0
        else:
            # Linear between LOW_VOL_PCT and HIGH_VOL_PCT.
            score = 1.0 - (pct - LOW_VOL_PCT) / (HIGH_VOL_PCT - LOW_VOL_PCT)

        regime = "low" if pct <= LOW_VOL_PCT else "high" if pct >= HIGH_VOL_PCT else "mid"

        return ConvictionComponent(
            score=score,
            weight=self.default_weight,
            evidence=(
                f"rvol_20d={current_rvol:.1%} (pct={pct:.0f}, regime={regime}) — "
                f"overnight-bias favorability {score:.2f}"
            ),
            features={
                "rvol_20d": round(current_rvol, 4),
                "rvol_pct": round(pct, 1),
            },
        )
