"""Narrative generator — persistence, idempotency, graceful degradation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.journal.narratives import NarrativeGenerator, TradeNarrative
from aitrade.journal.round_trips import ExitReason, PnlBucket, TradeRoundTrip
from aitrade.logging.trade_logger import TradeLogger

# --- helpers ----------------------------------------------------------------


def _make_trip(
    *,
    trade_id: str = "trade-1",
    symbol: str = "AAPL",
    entry_ts: datetime | None = None,
    exit_ts: datetime | None = None,
) -> TradeRoundTrip:
    entry = entry_ts or datetime(2024, 4, 20, 14, 30, tzinfo=UTC)
    exit_ = exit_ts or datetime(2024, 4, 22, 14, 30, tzinfo=UTC)
    return TradeRoundTrip(
        trade_id=trade_id,
        symbol=symbol,
        strategy_id="sma-cross",
        entry_event_id="ev-entry",
        exit_event_id="ev-exit",
        entry_ts=entry,
        exit_ts=exit_,
        entry_price=172.50,
        exit_price=175.80,
        qty=1.0,
        pnl_usd=3.30,
        pnl_pct=0.01913,
        holding_secs=int((exit_ - entry).total_seconds()),
        pnl_bucket=PnlBucket.WIN,
        exit_reason=ExitReason.TARGET_HIT,
        entry_thesis="3-day volume breakout",
        entry_catalyst="SPY bullish",
        entry_confidence=0.7,
        pattern_hits=["volume_breakout"],
        market_snapshot={"regime": "risk_on_low_vol"},
    )


def _settings_with_key(key: str = "test-key") -> Settings:
    return Settings(anthropic_api_key=SecretStr(key))


def _settings_no_key() -> Settings:
    return Settings(anthropic_api_key=SecretStr(""))


def _seed_round_trip(journal: TradeLogger, trip: TradeRoundTrip) -> None:
    """Insert a round_trip row directly; the FK requires it for the narrative."""
    # Ensure the round_trips table exists by importing the reconciler module.
    from aitrade.journal.round_trips import RoundTripReconciler

    RoundTripReconciler(journal)  # creates schema as a side-effect
    journal._conn.execute(  # noqa: SLF001
        "INSERT OR REPLACE INTO round_trips VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trip.trade_id,
            trip.symbol,
            trip.strategy_id,
            trip.entry_event_id,
            trip.exit_event_id,
            trip.entry_ts.isoformat(),
            trip.exit_ts.isoformat(),
            trip.entry_price,
            trip.exit_price,
            trip.qty,
            trip.pnl_usd,
            trip.pnl_pct,
            trip.holding_secs,
            trip.pnl_bucket.value,
            trip.exit_reason.value,
            trip.entry_thesis,
            trip.entry_catalyst,
            trip.entry_confidence,
            None,
            None,
        ),
    )


# --- fake anthropic streaming client ----------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeFinalMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_final_message(self) -> _FakeFinalMessage:
        return _FakeFinalMessage(self._text)


class _FakeMessages:
    def __init__(self, parent: _FakeAnthropic) -> None:
        self._parent = parent

    @contextmanager
    def stream(self, **kwargs: Any) -> Any:
        self._parent.calls += 1
        self._parent.last_kwargs = kwargs
        if self._parent.raise_exc is not None:
            raise self._parent.raise_exc
        yield _FakeStream(self._parent.text)


class _FakeAnthropic:
    def __init__(
        self, *, text: str = "stub narrative.", raise_exc: Exception | None = None
    ) -> None:
        self.text = text
        self.raise_exc = raise_exc
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None
        self.messages = _FakeMessages(self)


# --- tests ------------------------------------------------------------------


def test_narrative_persisted_with_mock_llm(tmp_path: Path) -> None:
    """Stub the anthropic client; narrative is returned and a row lands in SQLite."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    trip = _make_trip()
    _seed_round_trip(journal, trip)

    gen = NarrativeGenerator(journal, settings=_settings_with_key())
    fake = _FakeAnthropic(
        text=(
            "Bought AAPL at $172.50 on Apr 20. Closed at $175.80 Apr 22 for "
            "+1.91% ($3.30). Thesis confirmed. Held 2.0d."
        )
    )
    gen._client = fake  # type: ignore[assignment]

    out = gen.write_for(trip)

    assert out is not None
    assert isinstance(out, TradeNarrative)
    assert out.trade_id == trip.trade_id
    assert "AAPL" in out.narrative
    assert out.model == "claude-haiku-4-5"
    assert fake.calls == 1
    # The system prompt was sent with cache_control.
    assert fake.last_kwargs is not None
    system = fake.last_kwargs["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    # Row landed in the SQLite table.
    cur = journal._conn.execute(  # noqa: SLF001
        "SELECT trade_id, narrative, model FROM trade_narratives WHERE trade_id = ?",
        (trip.trade_id,),
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] == trip.trade_id
    assert row[1] == out.narrative
    assert row[2] == "claude-haiku-4-5"
    journal.close()


def test_narrative_idempotent(tmp_path: Path) -> None:
    """Calling write_for twice returns the same narrative; LLM is hit only once."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    trip = _make_trip()
    _seed_round_trip(journal, trip)

    gen = NarrativeGenerator(journal, settings=_settings_with_key())
    fake = _FakeAnthropic(text="first text")
    gen._client = fake  # type: ignore[assignment]

    first = gen.write_for(trip)
    # Change the fake so we can prove the second call doesn't hit it.
    fake.text = "second text"
    second = gen.write_for(trip)

    assert first is not None
    assert second is not None
    assert first.trade_id == second.trade_id
    assert first.narrative == second.narrative == "first text"
    assert fake.calls == 1
    journal.close()


def test_narrative_returns_none_on_llm_failure(tmp_path: Path) -> None:
    """LLM raises → write_for returns None; no row is persisted."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    trip = _make_trip()
    _seed_round_trip(journal, trip)

    gen = NarrativeGenerator(journal, settings=_settings_with_key())
    fake = _FakeAnthropic(raise_exc=RuntimeError("VPN dropped the connection"))
    gen._client = fake  # type: ignore[assignment]

    out = gen.write_for(trip)

    assert out is None
    cur = journal._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM trade_narratives WHERE trade_id = ?",
        (trip.trade_id,),
    )
    assert cur.fetchone()[0] == 0
    journal.close()


def test_narrative_returns_none_when_no_api_key(tmp_path: Path) -> None:
    """Empty API key → write_for returns None and never invokes the client."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    trip = _make_trip()
    _seed_round_trip(journal, trip)

    gen = NarrativeGenerator(journal, settings=_settings_no_key())
    fake = _FakeAnthropic()
    gen._client = fake  # type: ignore[assignment]

    out = gen.write_for(trip)

    assert out is None
    assert fake.calls == 0  # never invoked
    journal.close()


def test_get_returns_persisted_row(tmp_path: Path) -> None:
    """write_for then get returns the same TradeNarrative."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    trip = _make_trip()
    _seed_round_trip(journal, trip)

    gen = NarrativeGenerator(journal, settings=_settings_with_key())
    fake = _FakeAnthropic(text="persisted narrative.")
    gen._client = fake  # type: ignore[assignment]

    written = gen.write_for(trip)
    fetched = gen.get(trip.trade_id)

    assert written is not None
    assert fetched is not None
    assert fetched.trade_id == written.trade_id
    assert fetched.narrative == written.narrative
    assert fetched.model == written.model
    # created_at survives the round trip (compare ISO precision).
    assert fetched.created_at.isoformat() == written.created_at.isoformat()

    # Missing trade_id → None.
    assert gen.get("does-not-exist") is None
    journal.close()


def test_all_narratives_orders_by_created_desc(tmp_path: Path) -> None:
    """Two narratives written in order; all_narratives returns newest first."""
    journal = TradeLogger(log_dir=tmp_path, strategy_id="test")
    base = datetime(2024, 4, 20, tzinfo=UTC)
    trip_a = _make_trip(
        trade_id="trade-a",
        symbol="AAPL",
        entry_ts=base,
        exit_ts=base + timedelta(days=2),
    )
    trip_b = _make_trip(
        trade_id="trade-b",
        symbol="MSFT",
        entry_ts=base + timedelta(days=3),
        exit_ts=base + timedelta(days=4),
    )
    _seed_round_trip(journal, trip_a)
    _seed_round_trip(journal, trip_b)

    gen = NarrativeGenerator(journal, settings=_settings_with_key())
    fake = _FakeAnthropic(text="a-text")
    gen._client = fake  # type: ignore[assignment]

    first = gen.write_for(trip_a)
    assert first is not None
    # Force a strictly later created_at on the second row by overwriting
    # via a direct insert so test ordering is deterministic regardless of
    # clock resolution.
    later = (first.created_at + timedelta(seconds=5)).isoformat()
    journal._conn.execute(  # noqa: SLF001
        "INSERT INTO trade_narratives (trade_id, narrative, created_at, model) "
        "VALUES (?, ?, ?, ?)",
        (trip_b.trade_id, "b-text", later, "claude-haiku-4-5"),
    )

    rows = gen.all_narratives()
    assert [r.trade_id for r in rows] == [trip_b.trade_id, trip_a.trade_id]
    journal.close()
