"""Candidate-board scoring + ranking + position-fit bonus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.patterns.base import PatternSignal
from aitrade.patterns.board import build_board
from aitrade.strategy.signal import Direction


def _ticker(symbol: str, buzz: float, mentions: int = 5) -> DiscoveredTicker:
    return DiscoveredTicker(
        symbol=symbol,
        mention_count=mentions,
        source_weight=0.8,
        recency_minutes=10.0,
        buzz_score=buzz,
        discovered_at=datetime(2026, 4, 24, 14, 0, tzinfo=UTC),
        evidence=[f"snippet about {symbol}"],
    )


def _signal(symbol: str, name: str, score: float) -> PatternSignal:
    return PatternSignal(
        name=name,
        symbol=symbol,
        score=score,
        direction=Direction.LONG,
        timestamp=datetime(2026, 4, 24, 14, 0, tzinfo=UTC),
        evidence={"score": score},
    )


def test_empty_inputs_produce_empty_board() -> None:
    board = build_board([], {})
    assert board.candidates == []


def test_pure_noise_symbols_dropped() -> None:
    # Symbol present in patterns_by_symbol but with no signals AND no buzz.
    board = build_board([], {"NOISE": []})
    assert board.candidates == []


def test_higher_combined_score_ranks_first() -> None:
    discovered = [_ticker("AAA", buzz=2.0), _ticker("BBB", buzz=1.0)]
    patterns = {
        "AAA": [_signal("AAA", "volume_trend", 0.8)],
        "BBB": [_signal("BBB", "breakout", 0.6)],
    }
    board = build_board(discovered, patterns)
    assert [c.symbol for c in board.candidates] == ["AAA", "BBB"]
    assert board.candidates[0].combined_score > board.candidates[1].combined_score


def test_position_fit_penalizes_held_symbols() -> None:
    # Two symbols, identical raw scores. The held one should rank lower.
    discovered = [_ticker("HELD", buzz=1.0), _ticker("FRESH", buzz=1.0)]
    patterns = {
        "HELD": [_signal("HELD", "breakout", 0.5)],
        "FRESH": [_signal("FRESH", "breakout", 0.5)],
    }
    board = build_board(
        discovered,
        patterns,
        held_qty_by_symbol={"HELD": 100.0},
        last_price_by_symbol={"HELD": 100.0, "FRESH": 100.0},
        cash_available=10_000.0,
        target_notional=2_000.0,
    )
    held = next(c for c in board.candidates if c.symbol == "HELD")
    fresh = next(c for c in board.candidates if c.symbol == "FRESH")
    assert held.position_fit_bonus == -1.0
    assert fresh.position_fit_bonus == 1.0
    assert fresh.combined_score > held.combined_score


def test_z_score_centers_per_cycle() -> None:
    # Constant buzz across symbols → z=0 component, ranking depends on patterns.
    discovered = [_ticker("X", buzz=1.0), _ticker("Y", buzz=1.0)]
    patterns = {
        "X": [_signal("X", "volume_trend", 0.9)],
        "Y": [_signal("Y", "volume_trend", 0.1)],
    }
    board = build_board(discovered, patterns)
    syms = [c.symbol for c in board.candidates]
    assert syms[0] == "X"  # higher pattern wins when buzz is constant


def test_evidence_includes_z_components_and_pattern_dump() -> None:
    discovered = [_ticker("ZZZ", buzz=3.0, mentions=12)]
    patterns = {"ZZZ": [_signal("ZZZ", "pullback", 0.7)]}
    board = build_board(discovered, patterns)
    cand = board.candidates[0]
    assert "buzz_z" in cand.evidence
    assert "pattern_z" in cand.evidence
    assert "raw_buzz" in cand.evidence
    assert "raw_pattern" in cand.evidence
    assert cand.evidence["mention_count"] == 12
    assert cand.pattern_hits == ["pullback"]
    assert cand.evidence["patterns"][0]["name"] == "pullback"


def test_to_dict_is_json_friendly() -> None:
    discovered = [_ticker("AAA", buzz=1.0)]
    patterns = {"AAA": [_signal("AAA", "breakout", 0.5)]}
    board = build_board(discovered, patterns, now=datetime(2026, 4, 24, 14, 0, tzinfo=UTC))
    out = board.to_dict()
    import json

    json.dumps(out)  # must not raise
    assert out["count"] == 1
    assert out["candidates"][0]["symbol"] == "AAA"


def test_built_at_default_is_recent() -> None:
    board = build_board([_ticker("X", buzz=1.0)], {"X": [_signal("X", "breakout", 0.5)]})
    assert (datetime.now(UTC) - board.built_at) < timedelta(seconds=5)
