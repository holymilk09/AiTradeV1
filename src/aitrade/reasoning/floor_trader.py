"""Floor-trader Claude persona — one trade per cycle from a candidate board.

The :class:`FloorTraderReasoner` plays the role of a disciplined US-equities
desk trader. It receives a *board* of candidate tickers (each scored for
buzz + technical pattern hits) plus current account state and per-symbol
multi-timeframe indicators, and returns a single :class:`FloorTraderDecision`:
either *one* trade to take this cycle, or an explicit pass.

It is a sibling to :class:`aitrade.reasoning.claude_reasoner.ClaudeReasoner`
but does **not** satisfy the ``Reasoner`` Protocol — its decision schema is
different and it consumes a board, not a single symbol. Risk management,
sizing, and order routing remain the caller's responsibility.
"""

from __future__ import annotations

from time import monotonic

from loguru import logger
from pydantic import BaseModel, Field

from aitrade.config import Settings, get_settings
from aitrade.reasoning.decision import FloorTraderDecision

# Stable across requests — heads the system slot so prompt caching stays warm.
# noqa: E501 not used so the prose can wrap naturally.
FLOOR_TRADER_SYSTEM_PROMPT = (
    "You are a disciplined US-equities trader on a professional proprietary desk. "
    "Your job each cycle is to scan a small board of candidate tickers — each one "
    "already pre-filtered for unusual buzz and at least one firing technical pattern — "
    "and decide whether to take **one** trade. You may also pass entirely. You take at "
    "most one new position per cycle.\n\n"
    "You will receive, in JSON:\n"
    "- candidate_board: list of {symbol, combined_score, buzz_score, pattern_score, "
    "pattern_hits, evidence}\n"
    "- current_positions: open positions keyed by symbol, with qty and avg entry price\n"
    "- cash_available, buying_power, max_position_usd\n"
    "- market_snapshot: a JSON dump of the current market regime / breadth / volatility\n"
    "- multi_tf_snapshots: per-symbol indicator dump across multiple timeframes "
    "(e.g. 5m / 15m / 1h)\n"
    "- news_by_symbol: keyed by candidate symbol, recent headlines "
    "({headline, source, age_min, summary}). Use these as catalyst evidence. "
    "Empty = no recent news.\n"
    "- economic_calendar: upcoming US macro events (Fed/CPI/NFP/GDP/etc.) — "
    "each {date, event, prev, est, actual, impact}. Risk-off the day before "
    "high-impact events; sized smaller right before / after a release.\n"
    "- upcoming_earnings: earnings releases relevant to the board — each "
    "{date, symbol, time, eps_estimated, ...}. Earnings within 48h is a "
    "strong reason to pass or wait.\n"
    "- prior_experience: keyed by candidate symbol, a list of compact past round-trips "
    "the bot took on similar setups (same symbol / shared pattern / same regime). "
    "Each entry has {when, pattern, outcome, thesis, what_happened, market_then}. "
    "If empty for a symbol, you have no prior data on it — treat the trade as fresh.\n"
    "- deep_dig_by_symbol: keyed by candidate symbol, a structured second-opinion "
    "from a separate analyst persona. Each entry has {bull_case, bear_case, "
    "risk_factors, confidence_in_setup, recommendation, reasoning}. Use the "
    "bear_case + risk_factors as a sizing/pass check. The deep-dig analyst is "
    "advisory — you are still the decider. Empty = no second opinion this cycle.\n\n"
    "Hard rules — never break these:\n"
    "1. Never propose a trade whose target_notional_usd exceeds max_position_usd. Size "
    "smaller when confidence is lower; full size only when confluence is strong.\n"
    "2. Never chase a move that is already extended (e.g. stretched RSI on multiple "
    "timeframes, multi-ATR push above VWAP, a candidate that has already run far from "
    "its catalyst level).\n"
    "3. Prefer **confluence**: a real trade has pattern hit AND buzz AND room in the "
    "account AND agreement across at least two timeframes. A single firing pattern "
    "alone is not enough.\n"
    "4. If your overall confidence is below 0.6, set should_trade=false and put one "
    "sentence in reason_for_pass. Doing nothing is a valid, frequent answer.\n"
    "5. If a candidate symbol is already in current_positions at meaningful size, do "
    "not stack into it; prefer a different name or pass.\n"
    "6. v1 is long-only. Never pick direction=SHORT — use FLAT to close, LONG to enter.\n"
    "7. Never invent prices. Use only values that appear in the input.\n"
    "8. **Use prior_experience**. If past trades on this setup were mostly losses, "
    "lower confidence or pass; if they were mostly wins, you may upweight. "
    "Reference the experience in your reason_for_pass or thesis when it changes "
    "your mind. Empty = no signal in either direction.\n"
    "8a. **Cross-check deep_dig_by_symbol if present.** If the second-opinion "
    "analyst recommends 'pass' on your candidate, you must either justify the "
    "trade against their bear_case or pass. If they recommend 'watch' and your "
    "confidence is below 0.7, prefer waiting. Their risk_factors should "
    "inform sizing — when several risks fire, size smaller or pass.\n"
    "9. **Earnings within 48h** for the picked symbol → pass or size very small. "
    "Earnings can wipe out technical setups. Mention it in reason_for_pass.\n"
    "10. **High-impact macro events same-day** (FOMC, CPI, NFP) → reduce confidence "
    "across the board; prefer waiting until after the print.\n\n"
    "When you do trade, your output must include:\n"
    "- pick_symbol — the chosen ticker\n"
    "- direction — LONG (enter) or FLAT (close)\n"
    "- target_notional_usd — within the cap\n"
    "- confidence — 0..1\n"
    "- thesis — one sentence on the edge\n"
    "- catalyst — one sentence on *why now*\n"
    "- entry_price / stop_price / target_price — informational levels you would use\n\n"
    "When you pass, set should_trade=false and explain in reason_for_pass.\n\n"
    "Respond with a single JSON object matching the provided schema. No preamble, no "
    "markdown fences."
)


class FloorTraderInput(BaseModel):
    """Per-cycle context for the floor trader.

    The shape is deliberately permissive (``list[dict]`` / ``dict[str, dict]``)
    so upstream board builders, snapshot serializers, and indicator packs can
    evolve without breaking this contract.
    """

    candidate_board: list[dict[str, object]] = Field(
        description=(
            "Pre-scored candidates. Each entry needs at minimum symbol, "
            "combined_score, buzz_score, pattern_score, pattern_hits, evidence."
        ),
    )
    current_positions: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Keyed by symbol; values: {qty, avg_entry_price, ...}.",
    )
    cash_available: float = Field(ge=0.0)
    buying_power: float = Field(ge=0.0)
    max_position_usd: float = Field(gt=0.0)
    market_snapshot: dict[str, object] = Field(
        default_factory=dict,
        description="Serialized MarketSnapshot — regime, breadth, volatility, etc.",
    )
    multi_tf_snapshots: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Keyed by symbol; value is a multi-timeframe indicator dump.",
    )
    news_by_symbol: dict[str, list[dict[str, object]]] = Field(
        default_factory=dict,
        description=(
            "Phase 2: keyed by candidate symbol, a list of compact recent "
            "news headlines (each: {headline, source, age_min, summary}). "
            "Empty list = no recent news on this name."
        ),
    )
    economic_calendar: list[dict[str, object]] = Field(
        default_factory=list,
        description=(
            "Phase 2: upcoming US economic events for the day/week ahead "
            "(Fed/CPI/NFP/GDP/etc.) — each entry: "
            "{date, event, prev, est, actual, impact}. Empty if FMP key absent."
        ),
    )
    upcoming_earnings: list[dict[str, object]] = Field(
        default_factory=list,
        description=(
            "Phase 2: upcoming earnings releases relevant to the candidate "
            "board — each entry: {date, symbol, time, eps_estimated, ...}."
        ),
    )
    prior_experience: dict[str, list[dict[str, object]]] = Field(
        default_factory=dict,
        description=(
            "Phase 1.5 experience replay: keyed by candidate symbol, value is a "
            "list of compact past-round-trip dicts ({when, pattern, outcome, "
            "thesis, what_happened, market_then}). Empty list means no prior "
            "experience yet — the floor trader should treat the trade as fresh."
        ),
    )
    deep_dig_by_symbol: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description=(
            "Phase 4b deep-dig: keyed by top-K candidate symbol, value is a "
            "compact DeepDigVerdict dump ({bull_case, bear_case, risk_factors, "
            "confidence_in_setup, recommendation, reasoning}). Empty dict means "
            "no second opinion was generated this cycle."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}


def _render_floor_user_message(ctx: FloorTraderInput) -> str:
    """Render the per-cycle context as JSON. Variable part — not cached."""
    import json

    return json.dumps(
        {
            "candidate_board": ctx.candidate_board,
            "current_positions": ctx.current_positions,
            "account": {
                "cash_available": round(ctx.cash_available, 2),
                "buying_power": round(ctx.buying_power, 2),
                "max_position_usd": ctx.max_position_usd,
            },
            "market_snapshot": ctx.market_snapshot,
            "multi_tf_snapshots": ctx.multi_tf_snapshots,
            "news_by_symbol": ctx.news_by_symbol,
            "economic_calendar": ctx.economic_calendar,
            "upcoming_earnings": ctx.upcoming_earnings,
            "prior_experience": ctx.prior_experience,
            "deep_dig_by_symbol": ctx.deep_dig_by_symbol,
        },
        indent=2,
        default=str,
    )


class FloorTraderReasoner:
    """Claude floor-trader persona with prompt caching + a global rate limit.

    The rate limit is **global** (a single key) because each cycle only
    yields one decision, regardless of how many candidates are on the board.
    """

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
                "ANTHROPIC_API_KEY missing. Set it in .env or stub the client in tests."
            )
        self._min_gap_secs = min_gap_secs or self._settings.aitrade_reasoner_min_gap_secs
        self._model = model or self._settings.aitrade_reasoner_model
        self._last_call: float | None = None
        self._client: object | None = None  # lazy

    def _get_client(self) -> object:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value(),
            )
        return self._client

    def _rate_limited(self) -> bool:
        if self._last_call is None:
            return False
        return (monotonic() - self._last_call) < self._min_gap_secs

    def decide_board(self, ctx: FloorTraderInput) -> FloorTraderDecision | None:
        """Return one trade decision (or pass) for the cycle.

        Returns ``None`` only if the call was rate-limited or the SDK call
        failed — those are operational signals to skip the cycle silently.
        Otherwise returns a structured :class:`FloorTraderDecision` with
        ``should_trade`` set true or false.
        """
        if self._rate_limited():
            logger.debug("floor_trader rate-limited; skipping cycle")
            return None

        client = self._get_client()
        user_message = _render_floor_user_message(ctx)

        try:
            response = client.messages.parse(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": FLOOR_TRADER_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
                output_format=FloorTraderDecision,
            )
        except Exception as e:
            logger.warning("floor_trader call failed err={}", e)
            return None

        self._last_call = monotonic()

        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug(
                "floor_trader usage in={} out={} cache_read={} cache_write={}",
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
                getattr(usage, "cache_read_input_tokens", 0),
                getattr(usage, "cache_creation_input_tokens", 0),
            )

        decision: FloorTraderDecision | None = response.parsed_output
        if decision is None:
            logger.warning("floor_trader returned no parsed output")
            return None

        # Defense-in-depth: clamp to cap regardless of the prompt rule.
        if decision.target_notional_usd > ctx.max_position_usd:
            logger.warning(
                "floor_trader proposed notional {:.2f} > cap {:.2f}; clamping",
                decision.target_notional_usd,
                ctx.max_position_usd,
            )
            decision = decision.model_copy(
                update={"target_notional_usd": ctx.max_position_usd}
            )

        logger.info(
            "floor_trader decision trade={} symbol={} dir={} notional={:.2f} conf={:.2f}",
            decision.should_trade,
            decision.pick_symbol,
            decision.direction.value,
            decision.target_notional_usd,
            decision.confidence,
        )
        return decision


__all__ = [
    "FLOOR_TRADER_SYSTEM_PROMPT",
    "FloorTraderInput",
    "FloorTraderReasoner",
]
