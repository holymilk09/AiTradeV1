"""LLM-gated Bollinger reversion.

Wraps :class:`BollingerReversion` and routes every LONG entry through a
short Claude call that approves or rejects the trade. Exits are NEVER
gated (we don't trap a position because the LLM is unavailable).

This is the simplest non-trivial way to put the LLM brain *inside* a
strategy's decision loop. Each call sees:
  - the symbol
  - the most recent N closes (so it can read the chart context)
  - the bollinger setup: current close vs the band, z-score, mid
  - simple derived indicators we can compute from bars alone:
    20-day realized vol, 5-day return, RSI-14, distance from
    52-week high

The model returns a single-token answer ("APPROVE" or "REJECT") plus a
short reason. We treat APPROVE as a pass-through of the inner signal;
REJECT suppresses the entry.

Cost is bounded by entry frequency: bollinger fires ~6–12 times per
year on a single liquid name. At Haiku 4.5 prices (~$0.0005/call) the
whole 8-symbol panel backtest costs well under $1.

Why this lives next to bollinger_reversion. EXP-007 established that
bollinger is the only strategy with positive panel expectancy. The
question this strategy answers: does layering the LLM as a filter
improve net Sharpe / hit-rate / drawdown? If yes, we have evidence the
LLM is adding value beyond the deterministic signal. If no, the LLM
isn't beating a coin flip on this filter task — and we'd save the
tokens by leaving bollinger un-gated.

The decision and its reason are logged on every call so we can audit
the LLM's judgement after the fact (which trades it killed, which it
let through, and how those subsets performed).
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from aitrade.config import get_settings
from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.signal import Direction, Signal

_HISTORY_FOR_CONTEXT = 60  # last N closes shown to the LLM
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LlmGatedBollinger:
    symbol: str
    period: int = 20
    num_std: float = 1.5
    model: str = _DEFAULT_MODEL
    strategy_id: str = "llm_gated_bollinger"
    _inner: BollingerReversion = field(init=False)
    _closes: deque[float] = field(init=False)
    _client: Any = field(init=False, default=None)
    _api_key: str = field(init=False, default="")
    _last_decision: str = field(init=False, default="")
    n_approve: int = field(init=False, default=0)
    n_reject: int = field(init=False, default=0)
    n_error: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._inner = BollingerReversion(
            symbol=self.symbol, period=self.period, num_std=self.num_std
        )
        self._closes = deque(maxlen=_HISTORY_FOR_CONTEXT)
        # Prefer Settings (loads .env via pydantic-settings); fall back to
        # raw env for direct-shell invocations.
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
                    "ANTHROPIC_API_KEY missing — set it in .env to use llm_gated_bollinger"
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
            return signal  # exits pass unchanged

        if self._llm_approves(bar, signal):
            self.n_approve += 1
            return signal
        self.n_reject += 1
        return None

    def _llm_approves(self, bar: Bar, signal: Signal) -> bool:
        # Compute lightweight context the LLM can read.
        closes = list(self._closes)
        if len(closes) < 21:
            # Not enough context to evaluate; default-approve so we
            # don't artificially skip warmup-period trades.
            return True

        recent = closes[-21:]
        last_close = recent[-1]
        return_5d = (last_close / closes[-6] - 1.0) if len(closes) >= 6 else 0.0
        return_20d = (last_close / closes[-21] - 1.0) if len(closes) >= 21 else 0.0

        # 20d realized vol (annualized).
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

        # Distance from rolling-60d high.
        window_high = max(closes) if closes else last_close
        dist_from_high = (last_close / window_high - 1.0) * 100.0

        # Simple RSI-14.
        rsi = _rsi(closes[-15:]) if len(closes) >= 15 else 50.0

        prompt = (
            f"You are a contrarian mean-reversion specialist on a quant desk.\n\n"
            f"Bollinger mean-reversion strategy just signaled BUY on {self.symbol} "
            f"at ${last_close:.2f}.\n"
            f"Setup context: {signal.reason}\n\n"
            f"Recent context:\n"
            f"  - 5d return:       {return_5d:+.2%}\n"
            f"  - 20d return:      {return_20d:+.2%}\n"
            f"  - 20d realized vol: {rvol:.1%} (annualized)\n"
            f"  - Distance from 60d high: {dist_from_high:+.1f}%\n"
            f"  - RSI-14:          {rsi:.0f}\n"
            f"\n"
            f"IMPORTANT FRAMING. Bollinger mean-reversion entries are designed to "
            f"look ugly — closing below the lower band requires a sharp recent "
            f"drop. Negative 5d/20d returns and being meaningfully below the "
            f"60-day high are PRE-CONDITIONS for the trade, not red flags.\n"
            f"The empirical edge of this setup on liquid US large-caps is "
            f"+0.95 to +1.88 Sharpe across multiple symbols and walk-forward "
            f"folds (panel mean Sharpe +1.13 across an 8-name universe). You "
            f"add value by REJECTING the small fraction of setups that are "
            f"genuinely structurally broken, NOT by avoiding the standard "
            f"oversold-bounce pattern.\n"
            f"\n"
            f"REJECT ONLY for clear structural reasons:\n"
            f"  - 60d high distance worse than −30% (likely sustained bear)\n"
            f"  - 20d realized vol > 60% AND deeply negative (panic, not bounce)\n"
            f"  - RSI-14 already > 50 (not actually oversold — bad setup)\n"
            f"\n"
            f"Otherwise APPROVE. Default to APPROVE for the canonical case "
            f"(RSI 20–40, drawdown 5–20% from 60d high, vol 20–50%).\n"
            f"\n"
            f"Reply with exactly one of:\n"
            f"  APPROVE — <one short reason>\n"
            f"  REJECT  — <one short reason>"
        )

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip() if response.content else ""
        except Exception as e:
            self.n_error += 1
            logger.warning("llm_gated_bollinger: LLM call failed; default-approve. err={}", e)
            return True

        self._last_decision = text
        first_word = text.lstrip().split()[0].upper() if text else ""
        approve = first_word.startswith("APPROVE")
        logger.debug(
            "llm gate {} {} at ${:.2f}: {}",
            self.symbol,
            "APPROVE" if approve else "REJECT",
            last_close,
            text[:120],
        )
        return approve

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None


def _rsi(closes: list[float]) -> float:
    """Wilder's RSI-14 from a 15-close window. Returns 50.0 on bad input."""
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
