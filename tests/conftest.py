"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.data.models import Bar


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(
        alpaca_api_key=SecretStr("test_key"),
        alpaca_secret_key=SecretStr("test_secret"),
        alpaca_live_trade=False,
        aitrade_data_dir=tmp_path / "data",
        aitrade_log_dir=tmp_path / "logs",
        aitrade_max_position_usd=10_000.0,
        aitrade_max_daily_loss_usd=1_000.0,
        aitrade_max_orders_per_min=5,
    )


@pytest.fixture
def synthetic_bars() -> list[Bar]:
    """60 bars with a clear upward trend — triggers SMA crossover to LONG."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    price = 100.0
    for i in range(60):
        price = 100.0 if i < 20 else 100.0 + (i - 20) * 0.5
        bars.append(
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
    return bars
