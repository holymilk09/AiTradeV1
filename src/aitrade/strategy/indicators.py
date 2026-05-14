"""Technical indicators computed from recent bars.

Cheap, local, deterministic — run on every bar. The reasoner reads their values
along with position state when deciding. Pure functions; no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aitrade.data.models import Bar, Timeframe


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Every indicator we expose to the reasoner, at a single point in time."""

    symbol: str
    price: float
    sma_fast: float | None
    sma_slow: float | None
    ema_12: float | None
    ema_26: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    atr_14: float | None
    bull_trend: bool | None  # MA5 > MA10 > MA20 stack
    fast_slow_cross: str | None  # "golden" / "death" / None


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    k = 2.0 / (window + 1)
    ema_val = sum(values[:window]) / window
    for v in values[window:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def rsi(values: list[float], window: int = 14) -> float | None:
    if len(values) < window + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(-min(diff, 0.0))
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(bars: list[Bar], window: int = 14) -> float | None:
    if len(bars) < window + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(bars)):
        hi = bars[i].high
        lo = bars[i].low
        prev_close = bars[i - 1].close
        tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        trs.append(tr)
    return sum(trs[-window:]) / window


def typical_price(bar: Bar) -> float:
    """(high + low + close) / 3 — the conventional VWAP price input."""
    return (bar.high + bar.low + bar.close) / 3.0


def rolling_vwap(bars: list[Bar], window: int) -> float | None:
    """Volume-weighted typical-price average over the last ``window`` bars.

    Σ(typ_price × volume) / Σ(volume), trimmed to the rolling window.
    Returns None if fewer than ``window`` bars or total volume is zero.
    """
    if len(bars) < window:
        return None
    recent = bars[-window:]
    pv = 0.0
    v = 0.0
    for b in recent:
        pv += typical_price(b) * b.volume
        v += b.volume
    if v <= 0:
        return None
    return pv / v


def vwap_distance_pct(bars: list[Bar], window: int) -> float | None:
    """(close - vwap) / vwap, percent. >0 = above VWAP, <0 = below.

    Useful as a setup feature — close just crossed VWAP from below, etc.
    """
    if not bars:
        return None
    vwap = rolling_vwap(bars, window)
    if vwap is None or vwap <= 0:
        return None
    return (bars[-1].close / vwap - 1.0) * 100.0


def volume_ma(bars: list[Bar], window: int) -> float | None:
    """Simple average volume over the last ``window`` bars."""
    if len(bars) < window:
        return None
    return sum(b.volume for b in bars[-window:]) / window


def volume_ratio(bars: list[Bar], window: int) -> float | None:
    """Current bar volume / ``window``-bar average. >1 means above-average flow.

    A ratio of 1.5 means today is 50 % above its trailing average — the
    classic threshold for "institutional participation" in many trend
    setups.
    """
    if len(bars) < window + 1:
        return None
    avg = volume_ma(bars[:-1], window)
    if avg is None or avg <= 0:
        return None
    return bars[-1].volume / avg


def _macd(values: list[float]) -> tuple[float | None, float | None]:
    e12 = ema(values, 12)
    e26 = ema(values, 26)
    if e12 is None or e26 is None:
        return None, None
    macd_line = e12 - e26
    # signal = 9-period EMA of the MACD line — approximate with last 9 points
    if len(values) < 26 + 9:
        return macd_line, None
    macd_series: list[float] = []
    for end in range(26, len(values) + 1):
        window = values[:end]
        e12w = ema(window, 12)
        e26w = ema(window, 26)
        if e12w is not None and e26w is not None:
            macd_series.append(e12w - e26w)
    signal = ema(macd_series, 9)
    return macd_line, signal


def compute_snapshot(
    bars: list[Bar],
    *,
    fast_window: int = 10,
    slow_window: int = 30,
) -> IndicatorSnapshot:
    """Compute every indicator from the given bar history.

    Pass the full recent window (e.g. last 200 bars). Missing values
    (insufficient history) come back as None — the reasoner handles that.
    """
    if not bars:
        raise ValueError("need at least one bar")

    closes = [b.close for b in bars]
    last = bars[-1]
    fast = sma(closes, fast_window)
    slow = sma(closes, slow_window)

    # MA5 / MA10 / MA20 bullish stack — pattern from ZhuLinsen's daily_stock_analysis
    ma5 = sma(closes, 5)
    ma10 = sma(closes, 10)
    ma20 = sma(closes, 20)
    if ma5 is None or ma10 is None or ma20 is None:
        bull_trend: bool | None = None
    else:
        bull_trend = ma5 > ma10 > ma20

    # Fast/slow cross — look at previous bar vs current
    cross: str | None = None
    prev_fast = sma(closes[:-1], fast_window) if len(closes) >= slow_window + 1 else None
    prev_slow = sma(closes[:-1], slow_window) if len(closes) >= slow_window + 1 else None
    if (
        prev_fast is not None
        and prev_slow is not None
        and fast is not None
        and slow is not None
    ):
        if prev_fast <= prev_slow and fast > slow:
            cross = "golden"
        elif prev_fast >= prev_slow and fast < slow:
            cross = "death"

    macd_line, macd_signal = _macd(closes)

    return IndicatorSnapshot(
        symbol=last.symbol,
        price=last.close,
        sma_fast=fast,
        sma_slow=slow,
        ema_12=ema(closes, 12),
        ema_26=ema(closes, 26),
        rsi_14=rsi(closes, 14),
        macd=macd_line,
        macd_signal=macd_signal,
        atr_14=atr(bars, 14),
        bull_trend=bull_trend,
        fast_slow_cross=cross,
    )


@dataclass(frozen=True, slots=True)
class MultiTimeframeSnapshot:
    """Per-timeframe stack of :class:`IndicatorSnapshot`.

    Strategies that condition on multi-resolution context (e.g. trade the
    5-minute setup only when the 1-day stack is bullish) read the relevant
    timeframe via :meth:`tf`.
    """

    per_tf: dict[Timeframe, IndicatorSnapshot]

    def tf(self, timeframe: Timeframe) -> IndicatorSnapshot:
        """Return the snapshot for ``timeframe``; raises ``KeyError`` if absent."""

        return self.per_tf[timeframe]

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """JSON-friendly dump keyed by timeframe value (e.g. ``"1Day"``)."""

        return {
            tf.value: {
                "symbol": snap.symbol,
                "price": snap.price,
                "sma_fast": snap.sma_fast,
                "sma_slow": snap.sma_slow,
                "ema_12": snap.ema_12,
                "ema_26": snap.ema_26,
                "rsi_14": snap.rsi_14,
                "macd": snap.macd,
                "macd_signal": snap.macd_signal,
                "atr_14": snap.atr_14,
                "bull_trend": snap.bull_trend,
                "fast_slow_cross": snap.fast_slow_cross,
            }
            for tf, snap in self.per_tf.items()
        }


def compute_multi_tf_snapshot(
    bars_by_tf: dict[Timeframe, list[Bar]],
) -> MultiTimeframeSnapshot:
    """Compute an :class:`IndicatorSnapshot` per timeframe and bundle them.

    Each value in ``bars_by_tf`` must be a non-empty list of bars at that
    resolution. Raises ``ValueError`` for any empty list (an empty input
    means we have no data at all, which would silently mask a fetch bug).
    """

    per_tf: dict[Timeframe, IndicatorSnapshot] = {}
    for tf, bars in bars_by_tf.items():
        if not bars:
            raise ValueError(f"empty bar list for timeframe={tf.value}")
        per_tf[tf] = compute_snapshot(bars)
    return MultiTimeframeSnapshot(per_tf=per_tf)
