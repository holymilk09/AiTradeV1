"""Live-trade guardrail invariants."""

from __future__ import annotations

import pytest

from aitrade.config import LiveTradeNotUnlockedError, Settings, assert_paper_or_unlocked


def _settings(live: bool) -> Settings:
    return Settings(alpaca_live_trade=live)


def test_both_off_returns_paper_mode() -> None:
    assert assert_paper_or_unlocked(confirm_live=False, settings=_settings(False)) is False


def test_both_on_authorizes_live() -> None:
    assert assert_paper_or_unlocked(confirm_live=True, settings=_settings(True)) is True


def test_env_only_raises() -> None:
    with pytest.raises(LiveTradeNotUnlockedError, match="confirm_live=True"):
        assert_paper_or_unlocked(confirm_live=False, settings=_settings(True))


def test_call_site_only_raises() -> None:
    with pytest.raises(LiveTradeNotUnlockedError, match="ALPACA_LIVE_TRADE=true"):
        assert_paper_or_unlocked(confirm_live=True, settings=_settings(False))
