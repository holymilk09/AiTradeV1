"""AlpacaBroker.is_tradable handles the SDK's enum status field correctly.

Regression: ``str(AssetStatus.ACTIVE).lower()`` yields ``"assetstatus.active"``,
not ``"active"`` — which silently classified every real symbol as
non-tradable and ran the paper executor into a rejection-spam loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock

import pytest

from aitrade.brokers.alpaca import AlpacaBroker
from aitrade.config import Settings


class _AssetStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class _FakeAsset:
    tradable: bool
    status: _AssetStatus


@pytest.fixture
def settings() -> Settings:
    from pydantic import SecretStr

    return Settings(
        alpaca_api_key=SecretStr("k"),
        alpaca_secret_key=SecretStr("s"),
        alpaca_live_trade=False,
    )


def _broker_with_asset(settings: Settings, asset: object) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._settings = settings  # type: ignore[attr-defined]
    broker._client = MagicMock()  # type: ignore[attr-defined]
    broker._client.get_asset.return_value = asset  # type: ignore[attr-defined]
    return broker


def test_is_tradable_handles_enum_status(settings: Settings) -> None:
    asset = _FakeAsset(tradable=True, status=_AssetStatus.ACTIVE)
    broker = _broker_with_asset(settings, asset)
    assert broker.is_tradable("AAPL") is True


def test_is_tradable_rejects_inactive(settings: Settings) -> None:
    asset = _FakeAsset(tradable=True, status=_AssetStatus.INACTIVE)
    broker = _broker_with_asset(settings, asset)
    assert broker.is_tradable("DELISTED") is False


def test_is_tradable_rejects_when_tradable_flag_false(settings: Settings) -> None:
    asset = _FakeAsset(tradable=False, status=_AssetStatus.ACTIVE)
    broker = _broker_with_asset(settings, asset)
    assert broker.is_tradable("HALTED") is False


def test_is_tradable_accepts_plain_string_status(settings: Settings) -> None:
    """Defensive: if Alpaca changes the SDK and starts returning plain strings."""
    asset = _FakeAsset(tradable=True, status="active")  # type: ignore[arg-type]
    broker = _broker_with_asset(settings, asset)
    assert broker.is_tradable("AAPL") is True


def test_is_tradable_fails_closed_on_exception(settings: Settings) -> None:
    broker = _broker_with_asset(settings, asset=None)
    broker._client.get_asset.side_effect = RuntimeError("net down")  # type: ignore[attr-defined]
    assert broker.is_tradable("AAPL") is False
