"""Parquet cache for historical bars. Layout borrowed from jon-becker's prediction-market-analysis:

  data/bars/<timeframe>/<symbol>.parquet
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from aitrade.data.models import Timeframe


class BarCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: Timeframe) -> Path:
        d = self.root / "bars" / timeframe.value
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol.upper()}.parquet"

    def load(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame | None:
        p = self._path(symbol, timeframe)
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def save(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        p = self._path(symbol, timeframe)
        df = df.sort_index() if df.index.name == "timestamp" else df.sort_values("timestamp")
        df.to_parquet(p, index=True)
        logger.debug("cached {} rows to {}", len(df), p)

    def upsert(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> pd.DataFrame:
        existing = self.load(symbol, timeframe)
        if existing is None:
            merged = df
        else:
            merged = pd.concat([existing, df])
            merged = merged[~merged.index.duplicated(keep="last")]
        self.save(symbol, timeframe, merged)
        return merged


def utcnow() -> datetime:
    return datetime.now(UTC)
