"""Conviction types + scorers + calibration end-to-end."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from aitrade.data.models import Bar
from aitrade.signals import ConvictionComponent, composite_from_components
from aitrade.signals.calibration import build_components, collect_setups, fit_weights
from aitrade.signals.components import (
    LlmReasoningScorer,
    MlSignalScorer,
    PatternMatchScorer,
    QuantEdgeScorer,
    ScoringContext,
)
from aitrade.strategy.examples.sma_crossover import SmaCrossover
from aitrade.strategy.signal import Direction, Signal


def _bars(n: int = 200, slope: float = 0.05) -> list[Bar]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    out: list[Bar] = []
    for i in range(n):
        price = 100.0 + i * slope
        out.append(
            Bar(
                symbol="TEST",
                timestamp=start + timedelta(days=i),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1_000_000,
            )
        )
    return out


def test_composite_weighted_average() -> None:
    components = {
        "a": ConvictionComponent(score=1.0, weight=0.6, evidence=""),
        "b": ConvictionComponent(score=0.0, weight=0.4, evidence=""),
    }
    assert composite_from_components(components) == 0.6


def test_composite_zero_weights() -> None:
    components = {
        "a": ConvictionComponent(score=1.0, weight=0.0, evidence=""),
        "b": ConvictionComponent(score=1.0, weight=0.0, evidence=""),
    }
    assert composite_from_components(components) == 0.0


def test_quant_edge_scorer() -> None:
    ctx = ScoringContext(
        bars=[],
        signal=Signal(symbol="X", direction=Direction.LONG, strength=0.8, reason="r"),
    )
    out = QuantEdgeScorer().score(ctx)
    assert out.score == 0.8
    assert out.evidence == "r"
    assert out.features == {"strength": 0.8}


def test_quant_edge_no_signal() -> None:
    ctx = ScoringContext(bars=[], signal=None)
    out = QuantEdgeScorer().score(ctx)
    assert out.score == 0.0


def test_ml_signal_returns_finite_score() -> None:
    ctx = ScoringContext(bars=_bars(), signal=None)
    out = MlSignalScorer().score(ctx)
    assert 0.0 <= out.score <= 1.0
    assert set(out.features) == {"ret_5", "ret_20", "vol_20_z", "range_20_z"}


def test_ml_signal_insufficient_history() -> None:
    ctx = ScoringContext(bars=_bars(n=10), signal=None)
    out = MlSignalScorer().score(ctx)
    assert out.score == 0.0


def test_pattern_match_runs_detectors() -> None:
    ctx = ScoringContext(bars=_bars(), signal=None)
    out = PatternMatchScorer().score(ctx)
    assert 0.0 <= out.score <= 1.0
    # features dict has one entry per detector
    assert len(out.features) >= 1


def test_llm_reasoning_default_off() -> None:
    out = LlmReasoningScorer().score(ScoringContext(bars=[], signal=None))
    assert out.weight == 0.0  # off until M4


def test_collect_setups_and_fit_weights() -> None:
    # Oscillating price so SMA crossover flips repeatedly post-warmup.
    import math

    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="TEST",
            timestamp=start + timedelta(days=i),
            open=100 + 5 * math.sin(i / 8),
            high=100 + 5 * math.sin(i / 8) + 0.5,
            low=100 + 5 * math.sin(i / 8) - 0.5,
            close=100 + 5 * math.sin(i / 8),
            volume=1_000_000,
        )
        for i in range(400)
    ]
    scorers = [QuantEdgeScorer(), MlSignalScorer(), PatternMatchScorer()]
    setups = collect_setups(
        lambda sym: SmaCrossover(symbol=sym),
        bars,
        scorers,
        horizon=5,
        warmup=60,
    )
    assert not setups.empty
    assert {"fwd_return", "quant_edge", "ml_signal", "pattern_match"}.issubset(setups.columns)

    result = fit_weights(setups, [s.name for s in scorers])
    assert result.n_setups == len(setups)
    assert abs(sum(result.fitted_weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in result.fitted_weights.values())


def test_build_components_applies_overrides() -> None:
    bars = _bars()
    scorers = [QuantEdgeScorer(), MlSignalScorer()]
    ctx = ScoringContext(
        bars=bars,
        signal=Signal(symbol="TEST", direction=Direction.LONG, strength=0.5, reason="x"),
    )
    out = build_components(
        scorers,
        ctx,
        weight_overrides={"quant_edge": 0.7, "ml_signal": 0.3},
    )
    assert out["quant_edge"].weight == 0.7
    assert out["ml_signal"].weight == 0.3
    # Score unchanged by override.
    assert out["quant_edge"].score == 0.5


def test_fit_weights_empty_setups() -> None:
    result = fit_weights(pd.DataFrame(), ["quant_edge", "ml_signal"])
    assert result.n_setups == 0
    assert result.composite_ic == 0.0
