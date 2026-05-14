"""TrendHunter — deterministic multi-criteria trend-strength scorer.

What this is: a pure-function scanner that takes daily bars and emits a
single ``TrendScore`` summarizing how *cleanly trending* the symbol is
right now. It composes the well-known textbook trend criteria into one
0-1 number plus a direction label.

What this is *not*: a pattern detector. The other patterns
(``volume_trend``, ``breakout``, ``gap_and_go`` ...) look for specific
*shapes* — single-bar setups. The TrendHunter looks for sustained
*alignment*: MA stack, higher-highs/higher-lows, volume confirmation,
RSI in the trend zone, MACD positive and rising.

The score is deterministic — no LLM call — so it's free to compute on
every candidate every cycle. It feeds into :class:`Candidate.trend_score`
and the floor-trader's prompt is reinforced to lean toward high
trend-score names when buzz + pattern align.

Design notes:

* Each component contributes a ``[0, 1]`` sub-score; the final score is
  their **mean** with explicit weights. Equal weight is the default —
  every component lives in the same scale, so unweighted mean is honest.
* Direction is decided by majority sign across components (long bias if
  most are bullish, short bias if most are bearish, ``FLAT`` otherwise).
* Failure mode: insufficient bars → ``TrendScore(score=0.0, direction=FLAT)``
  with empty ``components``. Caller can drop these or treat them as "no
  trend signal" — same as a pattern that didn't fire.
* Thresholds are tunable via constructor args; defaults are pinned to
  textbook values, not free parameters waiting to be overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aitrade.data.models import Bar
from aitrade.strategy.indicators import _macd, ema, rsi, sma
from aitrade.strategy.signal import Direction


@dataclass(frozen=True, slots=True)
class TrendScore:
    """Composite trend strength for a single symbol at a single point in time.

    ``score`` is in ``[0, 1]``: 0 = no trend, 1 = textbook clean trend.
    ``direction`` says which way (LONG/SHORT/FLAT). ``components`` is the
    breakdown — useful for journaling and for the floor-trader to see
    *why* a name scored high (e.g. "MA stack good, volume weak").
    """

    score: float
    direction: Direction
    components: dict[str, float] = field(default_factory=dict)

    @property
    def is_strong(self) -> bool:
        """Quick gate the engine uses to flag clean trenders for the prompt."""
        return self.score >= 0.7

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "direction": self.direction.value,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "is_strong": self.is_strong,
        }


# Textbook defaults — change one and rerun the journal-stats backtest.
_DEFAULT_MA_FAST = 20
_DEFAULT_MA_MID = 50
_DEFAULT_MA_SLOW = 200
_DEFAULT_VOL_LOOKBACK = 20
_DEFAULT_VOL_MULT = 1.2  # recent vol > 20d avg × 1.2 → score = 1.0
_DEFAULT_RSI_LO = 40.0
_DEFAULT_RSI_HI = 70.0
_DEFAULT_HH_LOOKBACK = 20  # higher-highs / higher-lows window pairs
_DEFAULT_MIN_BARS = 60  # need enough history to compute SMA50 + recent context


class TrendHunter:
    """Score how cleanly a symbol is trending, by stacking deterministic criteria.

    Components scored (each [0, 1]):

    1. **ma_stack** — close > SMA20 > SMA50 (and bonus when also > SMA200)
    2. **higher_highs_lows** — recent N-day high > prior N-day high AND recent
       N-day low > prior N-day low (or both lower for downtrend)
    3. **volume_confirm** — recent close-day volume vs 20-day SMA volume
    4. **rsi_zone** — 1.0 inside the [40, 70] trend zone, scaled outside
    5. **macd_positive** — MACD > signal (or for downtrend: < signal) AND rising

    Direction: majority-sign rule across the components (each component
    knows whether it's bullish, bearish, or neutral).
    """

    def __init__(
        self,
        *,
        ma_fast: int = _DEFAULT_MA_FAST,
        ma_mid: int = _DEFAULT_MA_MID,
        ma_slow: int = _DEFAULT_MA_SLOW,
        vol_lookback: int = _DEFAULT_VOL_LOOKBACK,
        vol_mult: float = _DEFAULT_VOL_MULT,
        rsi_lo: float = _DEFAULT_RSI_LO,
        rsi_hi: float = _DEFAULT_RSI_HI,
        hh_lookback: int = _DEFAULT_HH_LOOKBACK,
        min_bars: int = _DEFAULT_MIN_BARS,
    ) -> None:
        self.ma_fast = ma_fast
        self.ma_mid = ma_mid
        self.ma_slow = ma_slow
        self.vol_lookback = vol_lookback
        self.vol_mult = vol_mult
        self.rsi_lo = rsi_lo
        self.rsi_hi = rsi_hi
        self.hh_lookback = hh_lookback
        self.min_bars = min_bars

    def compute(self, bars: list[Bar]) -> TrendScore:
        """Return a TrendScore for the given daily bars. Pure; no I/O."""
        if len(bars) < self.min_bars:
            return TrendScore(score=0.0, direction=Direction.FLAT, components={})

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        last_close = closes[-1]

        ma_score, ma_dir = self._ma_stack_score(closes, last_close)
        hh_score, hh_dir = self._hh_hl_score(bars)
        vol_score = self._volume_score(volumes)
        rsi_score = self._rsi_score(closes)
        macd_score, macd_dir = self._macd_score(closes)

        components = {
            "ma_stack": ma_score,
            "higher_highs_lows": hh_score,
            "volume_confirm": vol_score,
            "rsi_zone": rsi_score,
            "macd_positive": macd_score,
        }
        score = sum(components.values()) / len(components)

        # Direction by majority across the directional components. The
        # volume + rsi components are direction-neutral (they only add
        # *strength*), so they don't vote.
        directions = [ma_dir, hh_dir, macd_dir]
        bull_votes = sum(1 for d in directions if d is Direction.LONG)
        bear_votes = sum(1 for d in directions if d is Direction.SHORT)
        if bull_votes > bear_votes:
            direction = Direction.LONG
        elif bear_votes > bull_votes:
            direction = Direction.SHORT
        else:
            direction = Direction.FLAT

        return TrendScore(score=score, direction=direction, components=components)

    # ------ component scorers ------------------------------------------------

    def _ma_stack_score(
        self, closes: list[float], last_close: float
    ) -> tuple[float, Direction]:
        """1.0 when close > SMA fast > SMA mid > SMA slow (full bull stack);
        0.0 when fully inverted; partial otherwise."""
        sma_f = sma(closes, self.ma_fast)
        sma_m = sma(closes, self.ma_mid)
        sma_s = sma(closes, self.ma_slow)
        if sma_f is None or sma_m is None:
            return 0.0, Direction.FLAT
        # Full bull stack — close > fast > mid > slow (slow optional if absent).
        bull = last_close > sma_f > sma_m and (sma_s is None or sma_m > sma_s)
        bear = last_close < sma_f < sma_m and (sma_s is None or sma_m < sma_s)
        if bull:
            return 1.0, Direction.LONG
        if bear:
            return 1.0, Direction.SHORT
        # Partial credit: count how many of the inequalities hold in the bull
        # direction; fractional score keeps the gradient honest.
        bull_hits = (
            int(last_close > sma_f)
            + int(sma_f > sma_m)
            + (int(sma_m > sma_s) if sma_s is not None else 0)
        )
        denom = 3 if sma_s is not None else 2
        return bull_hits / denom, Direction.LONG if bull_hits >= denom - 1 else Direction.FLAT

    def _hh_hl_score(self, bars: list[Bar]) -> tuple[float, Direction]:
        """Higher-highs + higher-lows over the last N vs prior N bars."""
        n = self.hh_lookback
        if len(bars) < 2 * n:
            return 0.0, Direction.FLAT
        recent = bars[-n:]
        prior = bars[-2 * n : -n]
        recent_high = max(b.high for b in recent)
        prior_high = max(b.high for b in prior)
        recent_low = min(b.low for b in recent)
        prior_low = min(b.low for b in prior)
        hh = recent_high > prior_high
        hl = recent_low > prior_low
        lh = recent_high < prior_high
        ll = recent_low < prior_low
        if hh and hl:
            return 1.0, Direction.LONG
        if lh and ll:
            return 1.0, Direction.SHORT
        # Mixed — half credit, no directional vote.
        return 0.5 if (hh or hl or lh or ll) else 0.0, Direction.FLAT

    def _volume_score(self, volumes: list[float]) -> float:
        """Recent volume vs ``vol_lookback``-day avg. Saturates at vol_mult×."""
        if len(volumes) < self.vol_lookback + 1:
            return 0.0
        avg = sum(volumes[-self.vol_lookback - 1 : -1]) / self.vol_lookback
        if avg <= 0:
            return 0.0
        ratio = volumes[-1] / avg
        # Linear ramp 0..1 between 0 and vol_mult; clamp.
        if ratio <= 0:
            return 0.0
        return min(1.0, ratio / self.vol_mult)

    def _rsi_score(self, closes: list[float]) -> float:
        """1.0 inside ``[rsi_lo, rsi_hi]``; linear ramp outside down to 0."""
        r = rsi(closes, 14)
        if r is None:
            return 0.0
        if self.rsi_lo <= r <= self.rsi_hi:
            return 1.0
        # Outside the zone: 0.5 credit at ±10 RSI points away, 0.0 at ±20.
        gap = min(abs(r - self.rsi_lo), abs(r - self.rsi_hi))
        return max(0.0, 1.0 - gap / 20.0)

    def _macd_score(self, closes: list[float]) -> tuple[float, Direction]:
        """MACD line vs signal AND rising over the last 3 bars."""
        macd_line, signal = _macd(closes)
        if macd_line is None or signal is None:
            return 0.0, Direction.FLAT
        # Rising / falling judged by EMA12 - EMA26 trend on a small window.
        e12 = ema(closes, 12)
        e26 = ema(closes, 26)
        if e12 is None or e26 is None:
            return 0.0, Direction.FLAT
        prev = closes[:-3] if len(closes) > 30 else None
        if prev is None or len(prev) < 26:
            rising = macd_line > signal
            falling = macd_line < signal
        else:
            prev_e12 = ema(prev, 12)
            prev_e26 = ema(prev, 26)
            if prev_e12 is None or prev_e26 is None:
                rising = macd_line > signal
                falling = macd_line < signal
            else:
                prev_macd = prev_e12 - prev_e26
                rising = macd_line > signal and macd_line > prev_macd
                falling = macd_line < signal and macd_line < prev_macd
        if rising:
            return 1.0, Direction.LONG
        if falling:
            return 1.0, Direction.SHORT
        return 0.3, Direction.FLAT


__all__ = ["TrendHunter", "TrendScore"]
