"""LLM-scored Bollinger reversion — graded gate.

Probe results (EXP-011, see docs/RESEARCH.md) showed that:

  - Binary (approve/reject) LLM prompts are prompt-bias-dominated — same
    set of bars get 1/51 approves with a conservative prompt and 52/52
    with a contrarian one. The LLM follows the framing rather than
    grading the chart.
  - Graded (1-10) scoring with concrete criteria *does* differentiate.
    Score distributions match independent regime classification:
    AAPL/MSFT score mean ~8 σ ~1; MU/PLTR span 3-9 with σ > 2. The LLM
    correctly flags panic-vol falling-knife setups with low scores
    while approving textbook oversold dislocations with high scores.

This strategy applies that finding: every Bollinger LONG entry is
scored 1-10 by Claude Haiku against an explicit additive scoring
rubric, and only setups with ``score >= min_score`` are taken.

Exits never go through the gate (a position must always be allowed
to flatten, even if the LLM thinks the setup looks fine).

Default ``min_score=7``: empirically the cut-point above which only
canonical oversold setups land; below it sits the cluster of
panic/structural rejections we want to skip.

Note on JSON: the model is instructed to return one JSON object. We
extract the first ``{...}`` in the reply and parse it. Parse failures
default to ``score=5`` (= REJECT under min_score=7) and log a warning.
This is the right asymmetry for a quant: when the LLM is confused,
*pass* the trade rather than take it.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from aitrade.config import get_settings
from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.signal import Direction, Signal

_HISTORY_FOR_CONTEXT = 60
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LlmScoredBollinger:
    symbol: str
    period: int = 20
    num_std: float = 1.5
    min_score: int = 7
    model: str = _DEFAULT_MODEL
    strategy_id: str = "llm_scored_bollinger"
    _inner: BollingerReversion = field(init=False)
    _closes: deque[float] = field(init=False)
    _client: Any = field(init=False, default=None)
    _api_key: str = field(init=False, default="")
    _last_score: int = field(init=False, default=0)
    _last_reason: str = field(init=False, default="")
    n_approve: int = field(init=False, default=0)
    n_reject: int = field(init=False, default=0)
    n_error: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not 1 <= self.min_score <= 10:
            raise ValueError("min_score must be in [1, 10]")
        self._inner = BollingerReversion(
            symbol=self.symbol, period=self.period, num_std=self.num_std
        )
        self._closes = deque(maxlen=_HISTORY_FOR_CONTEXT)
        try:
            self._api_key = get_settings().anthropic_api_key.get_secret_value()
        except Exception:
            self._api_key = ""
        if not self._api_key:
            self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY missing — set it in .env to use llm_scored_bollinger"
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None
        self._closes.append(bar.close)
        signal = self._inner.on_bar(bar)
        if signal is None:
            return None
        if signal.direction is not Direction.LONG:
            return signal

        score, reason = self._score(bar, signal)
        self._last_score = score
        self._last_reason = reason

        if score >= self.min_score:
            self.n_approve += 1
            return signal
        self.n_reject += 1
        return None

    def _score(self, bar: Bar, signal: Signal) -> tuple[int, str]:
        closes = list(self._closes)
        if len(closes) < 21:
            return 5, "insufficient history"

        last_close = closes[-1]
        return_5d = last_close / closes[-6] - 1.0
        return_20d = last_close / closes[-21] - 1.0
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
        window_high = max(closes)
        dist_from_high = (last_close / window_high - 1.0) * 100.0
        rsi = _rsi(closes[-15:]) if len(closes) >= 15 else 50.0

        prompt = (
            f"You score Bollinger mean-reversion setups 1-10 for entry quality.\n\n"
            f"Setup: {self.symbol} at ${last_close:.2f}\n"
            f"  Bollinger reason: {signal.reason}\n"
            f"  5d return:       {return_5d:+.2%}\n"
            f"  20d return:      {return_20d:+.2%}\n"
            f"  20d realized vol: {rvol:.1%}\n"
            f"  Distance from 60d high: {dist_from_high:+.1f}%\n"
            f"  RSI-14:          {rsi:.0f}\n\n"
            f"Score 1-10 by adding/subtracting from baseline 5:\n"
            f"  +2 if RSI < 30                 (deeply oversold)\n"
            f"  +1 if 30 <= RSI < 40           (oversold)\n"
            f"  +1 if -25% < dist_high < -5%   (typical drawdown range)\n"
            f"  +1 if 15% < rvol < 50%         (normal vol regime)\n"
            f"  -2 if dist_high < -30%         (likely sustained downtrend)\n"
            f"  -2 if rvol > 70%               (panic / structural)\n"
            f"  -1 if RSI > 50                 (not actually oversold)\n\n"
            f"Apply all rules that match. Clamp result to 1..10.\n\n"
            f"Reply with ONE JSON object only:\n"
            f'  {{"score": <1-10>, "reason": "<one sentence>"}}\n'
        )

        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self.model,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip() if resp.content else ""
        except Exception as e:
            self.n_error += 1
            logger.warning("llm_scored_bollinger: LLM call failed; default-score=5. err={}", e)
            return 5, f"error: {e}"

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return 5, f"parse fail: {text[:80]}"
        try:
            parsed: dict[str, Any] = json.loads(m.group(0))
            s = int(parsed.get("score", 5))
            s = max(1, min(10, s))
            reason = str(parsed.get("reason", ""))[:200]
            return s, reason
        except Exception as e:
            return 5, f"parse fail: {e}; text={text[:80]}"

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None


def _rsi(closes: list[float]) -> float:
    if len(closes) < 2:
        return 50.0
    gains = 0.0
    losses = 0.0
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
