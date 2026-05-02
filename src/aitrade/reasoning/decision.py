"""Reasoner input and output schemas.

The LLM never sees raw Python objects — it sees a formatted prompt with
these fields and returns structured JSON that validates against
``ReasonerDecision``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from aitrade.strategy.indicators import IndicatorSnapshot
from aitrade.strategy.signal import Direction


class ReasonerInput(BaseModel):
    """Full context handed to the reasoner for one decision."""

    symbol: str
    indicators: IndicatorSnapshot
    current_position_qty: float = 0.0
    avg_entry_price: float | None = None
    cash_available: float
    buying_power: float
    max_position_usd: float
    recent_fills_summary: str = ""

    model_config = {"arbitrary_types_allowed": True}


class ReasonerDecision(BaseModel):
    """Structured decision returned by the LLM. Risk gate runs after this."""

    direction: Direction = Field(
        description=(
            "long: want to be long, flat: want no position, "
            "short: want to be short (if supported)"
        ),
    )
    target_notional_usd: float = Field(
        ge=0,
        description="Desired notional size in USD. 0 means close position.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0..1 how confident the reasoner is in this decision.",
    )
    reason: str = Field(
        max_length=800,
        description="One-to-three-sentence justification. Will be logged.",
    )
    should_trade: bool = Field(
        default=True,
        description=(
            "False = hold / do nothing this cycle. Use when the reasoner "
            "is uncertain or current position already matches intent."
        ),
    )


class FloorTraderDecision(BaseModel):
    """One-trade-per-cycle pick from the floor-trader reasoner.

    Unlike :class:`ReasonerDecision` (which decides for one pre-selected
    symbol), this is the output of a reasoner that surveys an entire
    candidate board and chooses at most one symbol to act on — or passes
    on the cycle entirely.
    """

    should_trade: bool = Field(
        description="False means pass this cycle. All other fields are advisory only.",
    )
    pick_symbol: str | None = Field(
        default=None,
        description="The single chosen symbol. Required when should_trade=true.",
    )
    direction: Direction = Field(
        default=Direction.FLAT,
        description=(
            "LONG to enter / hold long, FLAT to close. Phase 1 does not "
            "support shorts; SHORT is rejected upstream."
        ),
    )
    target_notional_usd: float = Field(
        ge=0.0,
        default=0.0,
        description="Desired notional size in USD. Clamped to the cap by the caller.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="0..1 conviction. Below 0.6 the reasoner should usually pass.",
    )
    thesis: str = Field(
        max_length=400,
        default="",
        description="One-sentence trade thesis. Logged on the round-trip.",
    )
    catalyst: str = Field(
        max_length=400,
        default="",
        description="Why now? The trigger that justifies acting this cycle.",
    )
    entry_price: float | None = Field(
        default=None,
        description="Suggested entry price (informational; risk gate sizes the order).",
    )
    stop_price: float | None = Field(
        default=None,
        description="Hard stop (informational in v1; used as a learning signal).",
    )
    target_price: float | None = Field(
        default=None,
        description="Take-profit target (informational in v1).",
    )
    reason_for_pass: str | None = Field(
        default=None,
        description="Required when should_trade=false — one sentence on why no trade.",
    )


class DeepDigVerdict(BaseModel):
    """Phase 4b: a structured second-opinion analysis on a single candidate.

    Produced by a Claude Haiku call run on the top-K board candidates each
    cycle. The verdict is fed back to the floor-trader as part of its
    context so the final pick reflects an explicit consideration of the
    bull case, the bear case, and concrete risk factors — not a single
    pass over the indicator stack.

    All fields cap on length to keep the prompt cheap and the floor-trader
    context readable. ``recommendation`` is advisory only — the floor-trader
    is still the single source of truth on whether to trade.
    """

    symbol: str = Field(description="The candidate symbol this verdict is about.")
    bull_case: str = Field(
        max_length=400,
        description=(
            "The strongest case FOR the trade in 1-2 sentences. Reference "
            "concrete signals from the input (pattern, news, regime). No "
            "filler like 'Stock looks good'."
        ),
    )
    bear_case: str = Field(
        max_length=400,
        description=(
            "The strongest case AGAINST the trade in 1-2 sentences. Reference "
            "concrete red flags (extension, headline risk, prior failures). "
            "Required even when bullish — a real desk always has a counter-view."
        ),
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Up to 5 short bullet-style risk factors the floor-trader should "
            "size around (e.g. 'Earnings in 3 days', 'RSI extended on 1H'). "
            "Each ≤ 80 chars."
        ),
    )
    confidence_in_setup: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0..1 confidence that this is a real edge, NOT confidence in "
            "direction. Below 0.5 means weak signal regardless of bull/bear."
        ),
    )
    recommendation: Literal["trade", "watch", "pass"] = Field(
        description=(
            "trade = the setup is strong enough to act on; "
            "watch = interesting but wait for a better trigger; "
            "pass = signal too weak or risks too high."
        ),
    )
    reasoning: str = Field(
        max_length=400,
        description=(
            "One short paragraph (3-4 sentences) tying the bull/bear/risks "
            "together into the recommendation. Plain prose, no markdown."
        ),
    )
