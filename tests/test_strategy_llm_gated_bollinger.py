"""LlmGatedBollinger — mocked-LLM unit tests.

The strategy class is thin glue around BollingerReversion (already
tested) and a single Anthropic SDK call. These tests cover the
*wrapper-specific* behavior (approve/reject paths, error fallback,
counter accounting) with the LLM mocked out so we never burn tokens
in CI.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aitrade.data.models import Bar
from aitrade.strategy.examples.llm_gated_bollinger import LlmGatedBollinger
from aitrade.strategy.signal import Direction


def _bar(i: int, close: float) -> Bar:
    return Bar(
        symbol="TEST",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
        open=close, high=close + 0.1, low=close - 0.1, close=close, volume=1_000_000,
    )


def _trending_then_drop(period: int = 20) -> list[Bar]:
    """Stable rise + a band-breach drop on the last bar — known entry trigger."""
    bars = [_bar(i, 100.0 + i * 0.05) for i in range(period)]
    drop_price = bars[-1].close * 0.95
    bars.append(_bar(period, drop_price))
    return bars


def test_approve_passes_signal_through(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    monkeypatch.setattr(strat, "_llm_approves", lambda *a, **k: True)
    sig = None
    for b in _trending_then_drop():
        sig = strat.on_bar(b)
    assert sig is not None
    assert sig.direction is Direction.LONG
    assert strat.n_approve == 1


def test_reject_suppresses_signal(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    monkeypatch.setattr(strat, "_llm_approves", lambda *a, **k: False)
    sig = None
    for b in _trending_then_drop():
        sig = strat.on_bar(b)
    assert sig is None
    assert strat.n_reject == 1


def test_exit_signal_always_passes(monkeypatch) -> None:
    """Exits (FLAT) never go through the LLM gate."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    monkeypatch.setattr(strat, "_llm_approves", lambda *a, **k: True)
    # Enter long first.
    for b in _trending_then_drop():
        strat.on_bar(b)
    assert strat._inner._last_direction is Direction.LONG
    # Build an exit bar — close back at or above the mid; LLM gate
    # should not be consulted (we monkey-patch it to a reject-only
    # function and confirm exit still passes).
    monkeypatch.setattr(strat, "_llm_approves", lambda *a, **k: False)
    cur_mid = sum(strat._inner._closes) / len(strat._inner._closes)
    exit_bar = Bar(
        symbol="TEST",
        timestamp=datetime(2024, 3, 1, tzinfo=UTC),
        open=cur_mid * 1.02, high=cur_mid * 1.03,
        low=cur_mid * 1.0, close=cur_mid * 1.02, volume=1_000_000,
    )
    sig = strat.on_bar(exit_bar)
    assert sig is not None
    assert sig.direction is Direction.FLAT


def test_default_approve_when_insufficient_history() -> None:
    """If the strategy fires before we have 21 bars of context, the gate
    should default-approve rather than error. Verified by calling
    ``_llm_approves`` directly with a tiny ``_closes`` deque."""
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    # Manually seed the outer history to < 21 bars and confirm the gate
    # short-circuits to True without needing the network.
    strat._closes.extend([100.0, 101.0, 102.0])
    from aitrade.strategy.signal import Signal
    fake_signal = Signal(symbol="TEST", direction=Direction.LONG, strength=0.5, reason="test")
    assert strat._llm_approves(_bar(0, 100.0), fake_signal) is True


def test_llm_error_defaults_to_approve(monkeypatch) -> None:
    """If the Anthropic call raises, we don't trap the user — default
    APPROVE so the strategy degrades gracefully when the network is flaky."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)

    class _BrokenClient:
        def __init__(self) -> None:
            self.messages = self
        def create(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("simulated network failure")

    monkeypatch.setattr(strat, "_get_client", _BrokenClient)
    sig = None
    for b in _trending_then_drop():
        sig = strat.on_bar(b)
    assert sig is not None  # default-approved
    assert strat.n_approve == 1
    assert strat.n_error == 1


def test_missing_api_key_raises_on_first_real_call(monkeypatch) -> None:
    """If the key is absent AND we'd actually call the LLM, raise."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    strat = LlmGatedBollinger(symbol="TEST", period=20, num_std=2.0)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        strat._get_client()
