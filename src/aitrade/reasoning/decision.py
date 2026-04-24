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
