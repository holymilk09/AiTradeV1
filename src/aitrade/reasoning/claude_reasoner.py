"""Claude-based reasoner.

Reads a ``ReasonerInput`` (indicators, position, account) and returns a
structured ``ReasonerDecision``. Prompt caching keeps the system prompt hot
so repeated calls are cheap. A per-symbol rate limiter bounds token spend.

The reasoner never executes orders — it only decides. The bot's risk gate
and executor handle the rest.
"""

from __future__ import annotations

from time import monotonic
from typing import Protocol

from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.reasoning.decision import ReasonerDecision, ReasonerInput
from aitrade.reasoning.prompts import SYSTEM_PROMPT, render_user_message


class Reasoner(Protocol):
    """Anything that turns context → decision. Makes the reasoner mockable in tests."""

    def decide(self, ctx: ReasonerInput) -> ReasonerDecision | None: ...


class ClaudeReasoner:
    """Claude Opus 4.7 reasoner with prompt caching + per-symbol rate limit."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        min_gap_secs: int | None = None,
        model: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.anthropic_api_key.get_secret_value():
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing. Set it in .env or use "
                "aitrade.reasoning.MockReasoner in tests."
            )
        self._min_gap_secs = min_gap_secs or self._settings.aitrade_reasoner_min_gap_secs
        self._model = model or self._settings.aitrade_reasoner_model
        self._last_call: dict[str, float] = {}
        self._client: object | None = None  # lazy

    def _get_client(self) -> object:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value(),
            )
        return self._client

    def _rate_limited(self, symbol: str) -> bool:
        last = self._last_call.get(symbol)
        if last is None:
            return False
        return (monotonic() - last) < self._min_gap_secs

    def decide(self, ctx: ReasonerInput) -> ReasonerDecision | None:
        if self._rate_limited(ctx.symbol):
            logger.debug("reasoner rate-limited for {}", ctx.symbol)
            return None

        client = self._get_client()
        user_message = render_user_message(ctx)

        try:
            response = client.messages.parse(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
                output_format=ReasonerDecision,
            )
        except Exception as e:
            logger.warning("reasoner call failed symbol={} err={}", ctx.symbol, e)
            return None

        self._last_call[ctx.symbol] = monotonic()

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug(
                "reasoner usage symbol={} in={} out={} cache_read={} cache_write={}",
                ctx.symbol,
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
            )

        decision: ReasonerDecision | None = response.parsed_output
        if decision is None:
            logger.warning("reasoner returned no parsed output symbol={}", ctx.symbol)
            return None

        # Enforce the hard cap a second time — defense in depth against
        # a reasoner that ignores the prompt rule.
        if decision.target_notional_usd > ctx.max_position_usd:
            logger.warning(
                "reasoner proposed notional {:.2f} > cap {:.2f}; clamping",
                decision.target_notional_usd,
                ctx.max_position_usd,
            )
            decision = decision.model_copy(
                update={"target_notional_usd": ctx.max_position_usd}
            )

        logger.info(
            "reasoner decision symbol={} dir={} notional={:.2f} conf={:.2f} trade={}",
            ctx.symbol,
            decision.direction.value,
            decision.target_notional_usd,
            decision.confidence,
            decision.should_trade,
        )
        return decision
