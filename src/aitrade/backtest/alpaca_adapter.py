"""Adapters between Alpaca bar dataframes and our Bar DTO / Nautilus bars."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC

import pandas as pd

from aitrade.data.models import Bar


def df_to_bars(df: pd.DataFrame, symbol: str) -> Iterator[Bar]:
    """Yield Bar objects from an alpaca-py historical bars dataframe.

    Expects a DatetimeIndex and columns: open, high, low, close, volume,
    optional trade_count, vwap.
    """
    if df.empty:
        return
    for ts, row in df.iterrows():
        ts_ = pd.Timestamp(ts)
        ts_utc = ts_.tz_convert(UTC) if ts_.tzinfo else ts_.tz_localize(UTC)
        yield Bar(
            symbol=symbol,
            timestamp=ts_utc.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
            trade_count=int(row.get("trade_count", 0) or 0),
            vwap=float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
        )
