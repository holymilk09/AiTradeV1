"""Reasoner input and output schemas.

The LLM never sees raw Python objects — it sees a formatted prompt with
these fields and returns structured JSON that validates against
``ReasonerDecision``.
"""

from __future__ import annotations

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
