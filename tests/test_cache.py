"""Parquet bar cache round-trip."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from aitrade.data.cache import BarCache
from aitrade.data.models import Timeframe


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cache = BarCache(tmp_path)
    idx = pd.date_range(datetime(2024, 1, 1, tzinfo=UTC), periods=10, freq="1D")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.5, "low": 0.5, "close": 1.2, "volume": 100}, index=idx
    )
    df.index.name = "timestamp"
    cache.save("AAPL", Timeframe.DAY_1, df)

    loaded = cache.load("AAPL", Timeframe.DAY_1)
    assert loaded is not None
    assert len(loaded) == 10


def test_upsert_deduplicates(tmp_path: Path) -> None:
    cache = BarCache(tmp_path)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df1 = pd.DataFrame(
        {"open": 1.0, "high": 1, "low": 1, "close": 1, "volume": 1},
        index=pd.date_range(base, periods=5, freq="1D"),
    )
    df1.index.name = "timestamp"
    df2 = pd.DataFrame(
        {"open": 2.0, "high": 2, "low": 2, "close": 2, "volume": 2},
        index=pd.date_range(base + timedelta(days=3), periods=5, freq="1D"),
    )
    df2.index.name = "timestamp"

    cache.upsert("AAPL", Timeframe.DAY_1, df1)
    merged = cache.upsert("AAPL", Timeframe.DAY_1, df2)
    assert len(merged) == 8
    # later write wins on overlapping index
    assert float(merged.loc[base + timedelta(days=3), "open"]) == 2.0
