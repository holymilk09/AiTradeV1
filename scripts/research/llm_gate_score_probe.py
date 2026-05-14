"""Probe v2: does the LLM differentiate setups when asked to SCORE them?

The binary-gate probe showed Haiku 4.5 follows prompt framing rather than
signal quality on the same setups: a conservative prompt rejected 50/51,
a contrarian prompt approved 52/52. Same bars. Same chart context. The
LLM isn't grading — it's enforcing instruction bias.

This probe asks for a 1–10 score with concrete criteria instead of a
yes/no. If the LLM is actually differentiating, scores will spread.
If it's still prompt-bias-dominated, scores will cluster.

Output: per-symbol distribution of scores across all Bollinger entries
in 2024-2026 daily. Mean, stdev, full set.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import deque
from datetime import UTC, datetime
from statistics import fmean, pstdev
from typing import Any

import anthropic

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Bar, Timeframe
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.signal import Direction

SYMBOLS = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "PLTR", "MSFT", "BA", "MU"]
MODEL = "claude-haiku-4-5-20251001"
HISTORY = 60

settings = get_settings()
api_key = settings.anthropic_api_key.get_secret_value()
client = anthropic.Anthropic(api_key=api_key)


def _rsi(closes: list[float]) -> float:
    if len(closes) < 2:
        return 50.0
    gains = losses = 0.0
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses += -d
    if losses == 0:
        return 100.0
    rs = (gains / 14) / (losses / 14)
    return 100.0 - (100.0 / (1.0 + rs))


def _features(closes: list[float]) -> dict[str, float]:
    last = closes[-1]
    r5 = last / closes[-6] - 1.0 if len(closes) >= 6 else 0.0
    r20 = last / closes[-21] - 1.0 if len(closes) >= 21 else 0.0
    rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(max(1, len(closes) - 20), len(closes))
    ]
    if len(rets) >= 2:
        mean_r = sum(rets) / len(rets)
        var_r = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        rvol = math.sqrt(var_r) * math.sqrt(252.0)
    else:
        rvol = 0.0
    high_60 = max(closes)
    dist_high = (last / high_60 - 1.0) * 100.0
    rsi_14 = _rsi(closes[-15:]) if len(closes) >= 15 else 50.0
    return {
        "r5": r5, "r20": r20, "rvol": rvol,
        "dist_high": dist_high, "rsi": rsi_14, "close": last,
    }


def _score(symbol: str, f: dict[str, float], reason: str) -> tuple[int, str]:
    prompt = (
        f"You score Bollinger mean-reversion setups 1–10 for entry quality.\n\n"
        f"Setup: {symbol} at ${f['close']:.2f}\n"
        f"  Bollinger reason: {reason}\n"
        f"  5d return:       {f['r5']:+.2%}\n"
        f"  20d return:      {f['r20']:+.2%}\n"
        f"  20d realized vol: {f['rvol']:.1%}\n"
        f"  Distance from 60d high: {f['dist_high']:+.1f}%\n"
        f"  RSI-14:          {f['rsi']:.0f}\n\n"
        f"Score 1–10 by adding/subtracting based on:\n"
        f"  +2 if RSI < 30                 (deeply oversold)\n"
        f"  +1 if 30 <= RSI < 40           (oversold)\n"
        f"  +1 if -25% < dist_high < -5%   (typical drawdown range)\n"
        f"  +1 if 15% < rvol < 50%         (normal vol regime)\n"
        f"  -2 if dist_high < -30%         (likely sustained downtrend)\n"
        f"  -2 if rvol > 70%               (panic / structural)\n"
        f"  -1 if RSI > 50                 (not actually oversold)\n\n"
        f"Start from baseline 5. Apply all rules that match. Clamp 1..10.\n\n"
        f"Reply with ONE JSON object only:\n"
        f'  {{"score": <1-10>, "reason": "<one-sentence why>"}}\n'
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip() if resp.content else ""
    # Extract JSON.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return 5, f"(parse fail: {text[:80]})"
    try:
        parsed: dict[str, Any] = json.loads(m.group(0))
        s = int(parsed.get("score", 5))
        s = max(1, min(10, s))
        return s, parsed.get("reason", "")[:120]
    except Exception:
        return 5, f"(parse fail: {text[:80]})"


def probe(sym: str) -> tuple[list[int], list[tuple[str, int, str]]]:
    data_client = AlpacaDataClient(settings)
    df = data_client.fetch_stock_bars(
        sym, Timeframe.DAY_1,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    if df.empty:
        return [], []
    bars: list[Bar] = list(df_to_bars(df, sym))

    inner = BollingerReversion(symbol=sym, period=20, num_std=1.5)
    closes_deque: deque[float] = deque(maxlen=HISTORY)
    scores: list[int] = []
    by_bar: list[tuple[str, int, str]] = []
    for bar in bars:
        closes_deque.append(bar.close)
        sig = inner.on_bar(bar)
        if sig is None or sig.direction is not Direction.LONG:
            continue
        if len(closes_deque) < 21:
            continue
        f = _features(list(closes_deque))
        s, reason = _score(sym, f, sig.reason)
        scores.append(s)
        by_bar.append((str(bar.timestamp.date()), s, reason))
    return scores, by_bar


for sym in SYMBOLS:
    scores, details = probe(sym)
    if not scores:
        print(f"\n=== {sym} === no entries")
        continue
    mean = fmean(scores)
    sd = pstdev(scores) if len(scores) > 1 else 0.0
    print(f"\n=== {sym} === n={len(scores)} mean={mean:.2f} std={sd:.2f} "
          f"min={min(scores)} max={max(scores)}")
    for ts, s, r in details:
        print(f"  {ts}  score={s:>2d}  {r}")
