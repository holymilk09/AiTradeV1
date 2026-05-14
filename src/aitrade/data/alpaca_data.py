"""Alpaca historical + streaming data access."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.data.cache import BarCache
from aitrade.data.models import Timeframe

_ALPACA_TF_MAP: dict[Timeframe, str] = {
    Timeframe.MIN_1: "1Min",
    Timeframe.MIN_5: "5Min",
    Timeframe.MIN_15: "15Min",
    Timeframe.HOUR_1: "1Hour",
    Timeframe.DAY_1: "1Day",
}


class AlpacaDataClient:
    """Thin wrapper over alpaca-py historical + stream clients with parquet caching."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._cache = BarCache(self._settings.aitrade_data_dir)
        self._hist_stock = None
        self._hist_crypto = None

    def _stock_client(self):  # type: ignore[no-untyped-def]
        if self._hist_stock is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._hist_stock = StockHistoricalDataClient(
                api_key=self._settings.alpaca_api_key.get_secret_value(),
                secret_key=self._settings.alpaca_secret_key.get_secret_value(),
            )
        return self._hist_stock

    def _crypto_client(self):  # type: ignore[no-untyped-def]
        if self._hist_crypto is None:
            from alpaca.data.historical import CryptoHistoricalDataClient

            self._hist_crypto = CryptoHistoricalDataClient()
        return self._hist_crypto

    def fetch_stock_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if use_cache:
            cached = self._cache.load(symbol, timeframe)
            if cached is not None and not cached.empty:
                mask = (cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))
                window = cached.loc[mask]
                if not window.empty:
                    logger.debug("cache hit {} {} rows={}", symbol, timeframe.value, len(window))
                    return window

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame as AlpacaTF
        from alpaca.data.timeframe import TimeFrameUnit

        tf_map = {
            Timeframe.MIN_1: AlpacaTF(1, TimeFrameUnit.Minute),
            Timeframe.MIN_5: AlpacaTF(5, TimeFrameUnit.Minute),
            Timeframe.MIN_15: AlpacaTF(15, TimeFrameUnit.Minute),
            Timeframe.HOUR_1: AlpacaTF(1, TimeFrameUnit.Hour),
            Timeframe.DAY_1: AlpacaTF(1, TimeFrameUnit.Day),
        }

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map[timeframe],
            start=start,
            end=end,
        )
        bars = self._stock_client().get_stock_bars(req)
        df = bars.df
        if df is None or df.empty:
            logger.warning("no bars returned for {} {}", symbol, timeframe.value)
            return pd.DataFrame()

        # alpaca-py returns MultiIndex (symbol, timestamp); drop symbol level
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        df.index.name = "timestamp"

        if use_cache:
            self._cache.upsert(symbol, timeframe, df)
        return df
