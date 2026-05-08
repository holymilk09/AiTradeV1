"""Candidate-board scoring + ranking + position-fit bonus."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.patterns.base import PatternSignal
from aitrade.patterns.board import build_board
from aitrade.patterns.trend_hunter import TrendScore
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


# ----- Phase 5 — trend score in the combined score ----------------------------


def _trend(score: float, direction: Direction = Direction.LONG) -> TrendScore:
    return TrendScore(
        score=score,
        direction=direction,
        components={"ma_stack": score, "higher_highs_lows": score},
    )


def test_trend_score_breaks_ties_when_buzz_and_pattern_equal() -> None:
    """Two candidates with identical buzz + pattern; the one with higher
    trend_score must rank first."""
    discovered = [_ticker("TRND", buzz=1.0), _ticker("CHOP", buzz=1.0)]
    patterns = {
        "TRND": [_signal("TRND", "breakout", 0.5)],
        "CHOP": [_signal("CHOP", "breakout", 0.5)],
    }
    trends = {
        "TRND": _trend(0.85, Direction.LONG),
        "CHOP": _trend(0.20, Direction.FLAT),
    }
    board = build_board(discovered, patterns, trend_by_symbol=trends)
    assert [c.symbol for c in board.candidates] == ["TRND", "CHOP"]
    trnd = next(c for c in board.candidates if c.symbol == "TRND")
    assert trnd.trend_score == 0.85
    assert trnd.trend_direction is Direction.LONG


def test_symbol_with_only_trend_signal_is_kept() -> None:
    """A symbol surfacing on the trend lane alone (no buzz, no patterns) still
    appears on the board — TrendHunter is a first-class lane."""
    trends = {"TRND": _trend(0.9, Direction.LONG)}
    board = build_board([], {}, trend_by_symbol=trends)
    assert len(board.candidates) == 1
    assert board.candidates[0].symbol == "TRND"
    assert board.candidates[0].trend_score == 0.9


def test_trend_evidence_components_serialized() -> None:
    discovered = [_ticker("AAA", buzz=1.0)]
    patterns = {"AAA": [_signal("AAA", "breakout", 0.5)]}
    trends = {"AAA": _trend(0.75, Direction.LONG)}
    board = build_board(discovered, patterns, trend_by_symbol=trends)
    cand = board.candidates[0]
    assert "trend_z" in cand.evidence
    assert "raw_trend" in cand.evidence
    assert cand.evidence["trend_components"] == {"ma_stack": 0.75, "higher_highs_lows": 0.75}
    assert cand.evidence["trend_is_strong"] is True


# ----- Phase 8 — correlation_penalty in combined_score ---------------------


def test_correlation_penalty_demotes_stacked_candidate() -> None:
    """Two candidates with identical raw scores; the one whose return
    series is highly correlated with a held position gets penalized
    and ranks below the uncorrelated peer."""
    discovered = [_ticker("STACKED", buzz=1.0), _ticker("FRESH", buzz=1.0)]
    patterns = {
        "STACKED": [_signal("STACKED", "breakout", 0.5)],
        "FRESH":   [_signal("FRESH",   "breakout", 0.5)],
    }
    penalty_map = {"STACKED": -0.5}  # caller computed this from corr matrix
    board = build_board(
        discovered,
        patterns,
        correlation_penalty_by_symbol=penalty_map,
    )
    syms = [c.symbol for c in board.candidates]
    assert syms == ["FRESH", "STACKED"]
    stacked = next(c for c in board.candidates if c.symbol == "STACKED")
    fresh = next(c for c in board.candidates if c.symbol == "FRESH")
    assert stacked.correlation_penalty == -0.5
    assert fresh.correlation_penalty == 0.0
    # Combined score must reflect the penalty.
    assert fresh.combined_score - stacked.combined_score == pytest.approx(0.5)
    # Penalty surfaces in evidence too — for the floor-trader prompt.
    assert stacked.evidence["correlation_penalty"] == -0.5
    assert "correlation_penalty" not in fresh.evidence


def test_correlation_penalty_serialized_in_to_dict() -> None:
    discovered = [_ticker("AAA", buzz=1.0)]
    patterns = {"AAA": [_signal("AAA", "breakout", 0.5)]}
    board = build_board(
        discovered, patterns, correlation_penalty_by_symbol={"AAA": -0.25}
    )
    out = board.candidates[0].to_dict()
    assert out["correlation_penalty"] == -0.25
