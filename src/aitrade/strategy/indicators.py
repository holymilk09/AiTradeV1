"""Technical indicators computed from recent bars.

Cheap, local, deterministic — run on every bar. The reasoner reads their values
along with position state when deciding. Pure functions; no state.
"""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.data.models import Bar


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
