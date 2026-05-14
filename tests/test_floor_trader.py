"""Floor-trader reasoner — clamping, rate limiting, pass-through."""

from __future__ import annotations

from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.reasoning.decision import FloorTraderDecision
from aitrade.reasoning.floor_trader import FloorTraderInput, FloorTraderReasoner
from aitrade.strategy.signal import Direction


def _ctx() -> FloorTraderInput:
    return FloorTraderInput(
        candidate_board=[
            {
                "symbol": "AAPL",
                "combined_score": 0.82,
                "buzz_score": 0.7,
                "pattern_score": 0.9,
                "pattern_hits": ["gap_and_go", "breakout"],
                "evidence": {"vwap_dist": 0.4},
            },
            {
                "symbol": "MSFT",
                "combined_score": 0.65,
                "buzz_score": 0.6,
                "pattern_score": 0.7,
                "pattern_hits": ["pullback"],
                "evidence": {"vwap_dist": 0.1},
            },
        ],
        current_positions={},
        cash_available=50_000.0,
        buying_power=100_000.0,
        max_position_usd=5_000.0,
        market_snapshot={"regime": "trend_up"},
        multi_tf_snapshots={
            "AAPL": {"5m": {"rsi": 55}, "1h": {"rsi": 58}},
            "MSFT": {"5m": {"rsi": 50}, "1h": {"rsi": 52}},
        },
    )


class _MockClient:
    """Mirrors the shape used by `tests/test_reasoner.py`: messages.parse(...)."""

    def __init__(self, decision: FloorTraderDecision) -> None:
        self.messages = self
        self._decision = decision
        self.calls = 0

    def parse(self, **_kwargs: object) -> object:
        self.calls += 1
        stub = type("R", (), {})()
        stub.parsed_output = self._decision
        stub.usage = None
        return stub


def _make_reasoner(decision: FloorTraderDecision, min_gap: int = 1) -> FloorTraderReasoner:
    settings = Settings(anthropic_api_key=SecretStr("test"))
    r = FloorTraderReasoner(settings=settings, min_gap_secs=min_gap)
    r._client = _MockClient(decision)  # type: ignore[assignment]
    return r


def test_floor_trader_returns_decision() -> None:
    decision = FloorTraderDecision(
        should_trade=True,
        pick_symbol="AAPL",
        direction=Direction.LONG,
        target_notional_usd=2_500.0,
        confidence=0.78,
        thesis="Confluence: breakout + buzz + bullish multi-tf.",
        catalyst="Fresh gap-and-go above prior day's high.",
        entry_price=190.0,
        stop_price=187.0,
        target_price=196.0,
    )
    r = _make_reasoner(decision)
    out = r.decide_board(_ctx())
    assert out is not None
    assert out.should_trade is True
    assert out.pick_symbol == "AAPL"
    assert out.direction is Direction.LONG
    assert out.target_notional_usd == 2_500.0


def test_floor_trader_clamps_to_cap() -> None:
    decision = FloorTraderDecision(
        should_trade=True,
        pick_symbol="AAPL",
        direction=Direction.LONG,
        target_notional_usd=999_999.0,  # absurd
        confidence=0.9,
        thesis="x",
        catalyst="y",
    )
    r = _make_reasoner(decision)
    out = r.decide_board(_ctx())
    assert out is not None
    assert out.target_notional_usd == 5_000.0


def test_floor_trader_rate_limited_second_call() -> None:
    decision = FloorTraderDecision(
        should_trade=False,
        reason_for_pass="Confluence not strong enough.",
    )
    r = _make_reasoner(decision, min_gap=60)
    first = r.decide_board(_ctx())
    second = r.decide_board(_ctx())
    assert first is not None
    assert second is None
    assert r._client.calls == 1  # type: ignore[union-attr]


def test_floor_trader_passes_through_no_trade() -> None:
    decision = FloorTraderDecision(
        should_trade=False,
        reason_for_pass="Market is choppy; no edge.",
    )
    r = _make_reasoner(decision)
    out = r.decide_board(_ctx())
    assert out is not None
    assert out.should_trade is False
    assert out.pick_symbol is None
    assert out.reason_for_pass == "Market is choppy; no edge."
