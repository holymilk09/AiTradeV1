"""Reasoner behavior — rate limit + cap clamping — with a mocked LLM."""

from __future__ import annotations

from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.reasoning.claude_reasoner import ClaudeReasoner
from aitrade.reasoning.decision import ReasonerDecision, ReasonerInput
from aitrade.strategy.indicators import IndicatorSnapshot
from aitrade.strategy.signal import Direction


def _ctx(symbol: str = "AAPL") -> ReasonerInput:
    ind = IndicatorSnapshot(
        symbol=symbol,
        price=150.0,
        sma_fast=149.0,
        sma_slow=148.0,
        ema_12=149.5,
        ema_26=148.5,
        rsi_14=55.0,
        macd=0.5,
        macd_signal=0.3,
        atr_14=2.0,
        bull_trend=True,
        fast_slow_cross="golden",
    )
    return ReasonerInput(
        symbol=symbol,
        indicators=ind,
        current_position_qty=0,
        cash_available=100_000.0,
        buying_power=200_000.0,
        max_position_usd=5_000.0,
    )


class _MockClient:
    def __init__(self, decision: ReasonerDecision) -> None:
        self.messages = self  # let .messages.parse() delegate here
        self._decision = decision
        self.calls = 0

    def parse(self, **_kwargs: object) -> object:  # noqa: D401
        self.calls += 1
        stub = type("R", (), {})()
        stub.parsed_output = self._decision
        stub.usage = None
        return stub


def _make_reasoner(decision: ReasonerDecision, min_gap: int = 1) -> ClaudeReasoner:
    settings = Settings(anthropic_api_key=SecretStr("test"))
    r = ClaudeReasoner(settings=settings, min_gap_secs=min_gap)
    r._client = _MockClient(decision)  # type: ignore[assignment]
    return r


def test_reasoner_returns_decision() -> None:
    decision = ReasonerDecision(
        direction=Direction.LONG,
        target_notional_usd=2_000.0,
        confidence=0.8,
        reason="Bullish stack + RSI neutral.",
    )
    r = _make_reasoner(decision)
    out = r.decide(_ctx())
    assert out is not None
    assert out.direction is Direction.LONG
    assert out.target_notional_usd == 2_000.0


def test_reasoner_clamps_to_cap() -> None:
    decision = ReasonerDecision(
        direction=Direction.LONG,
        target_notional_usd=999_999.0,  # way above cap
        confidence=0.9,
        reason="whatever",
    )
    r = _make_reasoner(decision)
    out = r.decide(_ctx())
    assert out is not None
    assert out.target_notional_usd == 5_000.0  # clamped to ctx.max_position_usd


def test_reasoner_rate_limits_second_call() -> None:
    decision = ReasonerDecision(
        direction=Direction.FLAT,
        target_notional_usd=0.0,
        confidence=0.5,
        reason="hold",
    )
    r = _make_reasoner(decision, min_gap=60)  # 60s cooldown
    first = r.decide(_ctx())
    second = r.decide(_ctx())
    assert first is not None
    assert second is None  # rate-limited
    assert r._client.calls == 1  # type: ignore[union-attr]
