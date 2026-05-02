"""Phase 4b: structured second-opinion (deep-dig) on top board candidates.

The :class:`DeepDigger` runs one Claude Haiku call per candidate and
returns a :class:`DeepDigVerdict` — bull case, bear case, risk factors,
recommendation. The floor trader sees these in its prompt under
``deep_dig_by_symbol`` and uses them as a sanity check on the buzz +
pattern board.

Why a single second-opinion call rather than a multi-persona swarm?

  * **Cost.** Three persona calls × top-3 candidates = 9 Haiku calls per
    cycle. A single call at top-3 is just 3 calls. At Haiku rates that's
    ~$0.003/cycle.
  * **Determinism.** A single structured output is easier to reason about
    than three free-form persona outputs the engine then has to fuse.
  * **Floor-trader is still the decider.** Adding more upstream voices
    risks the floor-trader rubber-stamping consensus rather than
    reasoning. One balanced "second analyst" is enough surface area.

The digger is **opt-in**, **best-effort** (every error path returns
``None`` and the floor-trader sees an empty ``deep_dig_by_symbol``
field), and runs only on the top-K candidates so per-cycle cost is
bounded regardless of how big the board is.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.reasoning.decision import DeepDigVerdict

# Stable system prompt — sits at position 0 of the request so prompt caching
# can amortize it across the cycle's top-K candidates and across cycles.
DEEP_DIG_SYSTEM_PROMPT = (
    "You are a senior US-equities analyst writing a one-page second opinion "
    "on a single ticker that another trader is considering. The trader has "
    "already pre-screened the symbol on buzz + pattern; your job is the "
    "*risk read* — bull case, bear case, key risks, plain-English verdict.\n\n"
    "You will receive, in JSON:\n"
    "- symbol\n"
    "- pattern_hits: which technical detectors fired (e.g. breakout, gap_and_go)\n"
    "- evidence: key indicator values + snippets that triggered the patterns\n"
    "- multi_tf: per-timeframe indicator stack (1D / 1H / 5m). Each TF has "
    "the relevant moving averages, RSI, MACD, ATR, and volume.\n"
    "- market_snapshot: current regime (SPY/QQQ/VIX, session time, regime tag)\n"
    "- news: recent headlines on this symbol (each {headline, source, age_min, "
    "summary}). Empty list = no recent news.\n"
    "- prior_experience: compact past round-trips on similar setups for this "
    "symbol — each {when, pattern, outcome, thesis}. Empty list = none.\n\n"
    "Hard rules:\n"
    "1. **Always produce a bear case**, even when bullish. A trader without a "
    "counter-view is one bad trade from blowing up.\n"
    "2. Reference *concrete* facts from the input (price levels, RSI values, "
    "headline summaries, prior outcomes). No vague phrasing like 'looks "
    "strong'. If multi-tf disagrees, name which TF disagrees.\n"
    "3. risk_factors are short bullets the trader will *size around*. Things "
    "like 'Earnings in 3 days', 'RSI 78 on 1H', 'Gap unfilled below'. Max 5.\n"
    "4. confidence_in_setup is 0..1 confidence that this is a real edge — "
    "NOT confidence in direction. Below 0.5 = weak signal regardless of "
    "bull/bear lean.\n"
    "5. recommendation: 'trade' (act now), 'watch' (interesting, wait for a "
    "better trigger), 'pass' (signal too weak or risks too high).\n"
    "6. Plain prose, no markdown, no emojis, no preambles. Match the schema.\n\n"
    "You are advisory. The trader makes the final call."
)


def _json_default(obj: Any) -> Any:
    """Coerce dataclasses, datetimes, and enums to JSON-serializable forms."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


class DeepDigger:
    """Run a single structured Haiku call per candidate symbol.

    Parameters
    ----------
    settings:
        Reads ``ANTHROPIC_API_KEY`` and the reasoner model. Tests inject a
        ``Settings`` with a stub key + replace ``_client`` directly.
    model:
        Override the Haiku model id. Defaults to ``claude-haiku-4-5``;
        Haiku is cheap, fast, and structurally on-spec for short structured
        outputs.
    max_top_k:
        Hard cap on how many candidates the engine should dig per cycle.
        Exposed here as a hint; the engine still controls the actual top-K.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model: str = "claude-haiku-4-5",
        max_top_k: int = 3,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.anthropic_api_key.get_secret_value():
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing. Set it in .env or stub the client in tests."
            )
        self._model = model
        self.max_top_k = max(1, max_top_k)
        self._client: object | None = None  # lazy

    def _get_client(self) -> object:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value(),
            )
        return self._client

    def dig(
        self,
        *,
        symbol: str,
        pattern_hits: list[str],
        evidence: list[str] | dict[str, object],
        multi_tf: dict[str, object],
        market_snapshot: dict[str, object],
        news: list[dict[str, object]] | None = None,
        prior_experience: list[dict[str, object]] | None = None,
    ) -> DeepDigVerdict | None:
        """Return a structured second-opinion verdict, or ``None`` on failure.

        Failure modes that yield ``None``:

        * The SDK call raises (network, rate limit, schema-validation error).
        * The model returns no parsed output.
        * The returned verdict's ``symbol`` doesn't match — defends against
          a confused output that mixed up the candidate.

        The caller (``engine_runner``) treats ``None`` as "this candidate
        has no second opinion this cycle" and continues the cycle.
        """
        client = self._get_client()
        payload = {
            "symbol": symbol,
            "pattern_hits": pattern_hits,
            "evidence": evidence,
            "multi_tf": multi_tf,
            "market_snapshot": market_snapshot,
            "news": news or [],
            "prior_experience": prior_experience or [],
        }
        # sort_keys=True keeps the rendered prompt byte-stable so the cache
        # can hit across symbols whose payloads only differ in field order.
        user_message = json.dumps(
            payload, default=_json_default, indent=2, sort_keys=True
        )

        try:
            response = client.messages.parse(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=600,
                system=[
                    {
                        "type": "text",
                        "text": DEEP_DIG_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
                output_format=DeepDigVerdict,
            )
        except Exception as e:
            logger.warning(
                "deep_dig call failed symbol={} err_type={} err={}",
                symbol,
                type(e).__name__,
                e,
            )
            return None

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug(
                "deep_dig usage symbol={} in={} out={} cache_read={} cache_write={}",
                symbol,
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
            )

        verdict: DeepDigVerdict | None = response.parsed_output
        if verdict is None:
            logger.warning("deep_dig returned no parsed output symbol={}", symbol)
            return None

        # Defense-in-depth: model occasionally substitutes the wrong symbol
        # in the output. If so, drop the verdict — better to skip than to
        # show the floor-trader a verdict for the wrong ticker.
        if verdict.symbol.upper() != symbol.upper():
            logger.warning(
                "deep_dig symbol mismatch wanted={} got={}; dropping",
                symbol,
                verdict.symbol,
            )
            return None

        logger.info(
            "deep_dig symbol={} rec={} conf={:.2f}",
            verdict.symbol,
            verdict.recommendation,
            verdict.confidence_in_setup,
        )
        return verdict


__all__ = [
    "DEEP_DIG_SYSTEM_PROMPT",
    "DeepDigger",
]
