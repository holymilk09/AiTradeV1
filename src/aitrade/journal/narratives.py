"""Per-round-trip narrative generator.

Turns each closed :class:`TradeRoundTrip` into a short, plain-English
post-mortem written by Claude Haiku. Narratives are persisted alongside
the journal so the same trade-id never gets re-summarised.

This layer is intentionally **best-effort**: if the API key is missing or
the LLM call fails, :meth:`NarrativeGenerator.write_for` returns ``None``
and logs a warning. It must never raise into a hot trading loop.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.journal.round_trips import TradeRoundTrip
from aitrade.logging.trade_logger import TradeLogger

# --- SQL ---------------------------------------------------------------------

_NARRATIVES_SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_narratives (
    trade_id TEXT PRIMARY KEY REFERENCES round_trips(trade_id),
    narrative TEXT NOT NULL,
    created_at TEXT NOT NULL,
    model TEXT NOT NULL
);
"""


# --- public dataclass --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeNarrative:
    """A persisted human-readable post-mortem of one closed round trip."""

    trade_id: str
    narrative: str
    created_at: datetime
    model: str


# --- prompts -----------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior trading-desk analyst writing one-paragraph post-mortems "
    "for closed trades.\n\n"
    "Your job: turn a single closed round-trip into a 3-4 sentence "
    "plain-English review.\n\n"
    "Voice: factual, neutral, direct. Like a desk-lead writing a one-line "
    "trade review on the way out the door.\n\n"
    "Rules:\n"
    "- 3-4 sentences. ~60-90 words. No more.\n"
    "- Cover: what was traded (symbol, dates), entry context (thesis, "
    "catalyst, market regime), outcome (P&L in USD and %, exit reason), "
    "and one line of judgment (\"thesis confirmed\", \"stop hit before "
    "pattern played out\", etc.).\n"
    "- Use concrete numbers from the JSON. NEVER invent prices, dates, or "
    "signals.\n"
    "- Plain prose only. No emojis. No markdown. No headings. No bullets. "
    "No preamble like \"Here is the summary\".\n"
    "- Refer to dates in human form (e.g. \"Apr 20\") and holding period "
    "in human-friendly units (e.g. \"2.0d\", \"45m\", \"3.5h\").\n"
    "- Past tense throughout. Speak as the desk reviewing a trade that "
    "already closed."
)


# --- helpers -----------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _humanize_holding(secs: int) -> str:
    if secs < 60:
        return f"{secs}s"
    if secs < 3_600:
        return f"{secs / 60:.0f}m"
    if secs < 86_400:
        return f"{secs / 3_600:.1f}h"
    return f"{secs / 86_400:.1f}d"


def _trip_to_payload(trip: TradeRoundTrip) -> dict[str, Any]:
    """Compact JSON shape passed to the LLM. Stable order for cache friendliness."""
    return {
        "symbol": trip.symbol,
        "strategy_id": trip.strategy_id,
        "entry_date": trip.entry_ts.strftime("%Y-%m-%d"),
        "exit_date": trip.exit_ts.strftime("%Y-%m-%d"),
        "entry_ts": trip.entry_ts,
        "exit_ts": trip.exit_ts,
        "entry_price": round(trip.entry_price, 4),
        "exit_price": round(trip.exit_price, 4),
        "qty": trip.qty,
        "pnl_usd": round(trip.pnl_usd, 2),
        "pnl_pct": round(trip.pnl_pct * 100, 2),
        "holding_period": _humanize_holding(trip.holding_secs),
        "holding_secs": trip.holding_secs,
        "pnl_bucket": trip.pnl_bucket,
        "exit_reason": trip.exit_reason,
        "entry_thesis": trip.entry_thesis,
        "entry_catalyst": trip.entry_catalyst,
        "entry_confidence": trip.entry_confidence,
        "pattern_hits": list(trip.pattern_hits),
        "market_snapshot_at_entry": trip.market_snapshot,
    }


def _row_to_narrative(row: tuple[Any, ...]) -> TradeNarrative:
    return TradeNarrative(
        trade_id=row[0],
        narrative=row[1],
        created_at=datetime.fromisoformat(row[2]),
        model=row[3],
    )


# --- generator ---------------------------------------------------------------


class NarrativeGenerator:
    """Generates and persists short post-mortems for closed round-trips."""

    def __init__(
        self,
        journal: TradeLogger,
        *,
        settings: Settings | None = None,
        model: str = "claude-haiku-4-5",
    ) -> None:
        self._journal = journal
        self._conn = journal._conn  # noqa: SLF001 — same DB file, single connection
        self._conn.executescript(_NARRATIVES_SCHEMA)
        self._settings = settings or get_settings()
        self._model = model
        self._client: object | None = None  # lazy

    # ----- public API -------------------------------------------------------

    def write_for(self, trip: TradeRoundTrip) -> TradeNarrative | None:
        """Generate, persist, and return a narrative for one round-trip.

        Idempotent: if a row already exists for ``trip.trade_id`` it is
        returned directly without an LLM call. Returns ``None`` if no API
        key is configured or the LLM call fails for any reason.
        """
        existing = self.get(trip.trade_id)
        if existing is not None:
            return existing

        if not self._settings.anthropic_api_key.get_secret_value():
            logger.warning(
                "narrative skipped trade_id={} reason=no_api_key", trip.trade_id
            )
            return None

        try:
            text = self._call_llm(trip)
        except Exception as e:
            logger.warning(
                "narrative LLM call failed trade_id={} err_type={} err={}",
                trip.trade_id,
                type(e).__name__,
                e,
            )
            return None

        if not text:
            logger.warning("narrative LLM returned empty text trade_id={}", trip.trade_id)
            return None

        narrative = TradeNarrative(
            trade_id=trip.trade_id,
            narrative=text,
            created_at=datetime.now(UTC),
            model=self._model,
        )
        self._persist(narrative)
        return narrative

    def get(self, trade_id: str) -> TradeNarrative | None:
        """Read a previously-persisted narrative by trade_id."""
        cur = self._conn.execute(
            "SELECT trade_id, narrative, created_at, model "
            "FROM trade_narratives WHERE trade_id = ?",
            (trade_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_narrative(row)

    def all_narratives(self) -> list[TradeNarrative]:
        """Return every persisted narrative, newest first."""
        cur = self._conn.execute(
            "SELECT trade_id, narrative, created_at, model "
            "FROM trade_narratives ORDER BY created_at DESC, trade_id DESC"
        )
        return [_row_to_narrative(row) for row in cur.fetchall()]

    # ----- internals --------------------------------------------------------

    def _get_client(self) -> object:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key.get_secret_value(),
            )
        return self._client

    def _call_llm(self, trip: TradeRoundTrip) -> str:
        """Stream the LLM response and return the accumulated text.

        Streaming is used (with ``get_final_message``) because non-streaming
        calls can be dropped by intermediate VPNs on long-ish responses.
        """
        client = self._get_client()
        user_message = json.dumps(
            _trip_to_payload(trip), default=_json_default, indent=2
        )

        with client.messages.stream(  # type: ignore[attr-defined]
            model=self._model,
            max_tokens=400,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            final = stream.get_final_message()

        # Accumulate any text blocks in the response.
        chunks: list[str] = []
        for block in getattr(final, "content", []) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks).strip()

    def _persist(self, narrative: TradeNarrative) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO trade_narratives "
                "(trade_id, narrative, created_at, model) VALUES (?, ?, ?, ?)",
                (
                    narrative.trade_id,
                    narrative.narrative,
                    narrative.created_at.isoformat(),
                    narrative.model,
                ),
            )
        except sqlite3.Error as e:
            logger.warning(
                "narrative persist failed trade_id={} err={}",
                narrative.trade_id,
                e,
            )


__all__ = [
    "NarrativeGenerator",
    "TradeNarrative",
]
