"""Ticker extraction + Alpaca tradable-universe loader.

The extractor runs two passes over free-form text:

* **Cashtag form** (``$AAPL``) — explicit market intent. Bypasses the blacklist;
  only the valid-tickers filter applies.
* **Contextual form** (``AAPL stock`` / ``TSLA calls``) — heuristic. The blacklist
  removes common English words that happen to match the regex.

Both passes are then intersected with ``valid_tickers`` (typically the Alpaca
tradable equity universe loaded by :func:`load_alpaca_active_equities`).
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from aitrade.config import Settings


_CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
_CONTEXTUAL_RE = re.compile(
    r"\b([A-Z]{2,5})\b(?=\s*(?:stock|shares|calls|puts|rally|surge|crash|dump))"
)

_DEFAULT_BLACKLIST: frozenset[str] = frozenset(
    {
        "I", "A", "T", "IT", "AT", "OR", "BE", "DO", "GO", "NO",
        "SO", "TV", "US", "ON", "AM", "PM", "UP", "OK",
    }
)

# 24h on-disk cache for the Alpaca asset list — refreshing it on every call is
# wasteful (the universe changes slowly) and adds a network dependency to tests.
_ASSET_CACHE_FILE = "alpaca_assets.json"
_ASSET_CACHE_TTL_SECS = 24 * 60 * 60


class TickerExtractor:
    """Extract candidate tickers from free-form text and validate against a universe.

    Parameters
    ----------
    valid_tickers:
        The set of symbols considered tradable (e.g. Alpaca active US equities).
        Anything outside this set is dropped, regardless of how it was matched.
    blacklist:
        Common English words to ignore in *contextual* matches. Cashtag matches
        bypass this list because ``$XYZ`` is unambiguous market intent. Defaults
        to a small set of two-letter words that frequently collide with tickers.
    """

    def __init__(
        self,
        valid_tickers: set[str],
        blacklist: set[str] | None = None,
    ) -> None:
        self._valid = {t.upper() for t in valid_tickers}
        self._blacklist = (
            {t.upper() for t in blacklist} if blacklist is not None else set(_DEFAULT_BLACKLIST)
        )

    def extract(self, text: str) -> list[tuple[str, int]]:
        """Return ``[(ticker, mention_count), ...]`` sorted desc by count.

        Cashtag mentions and contextual mentions of the same symbol are combined
        into a single count.
        """
        counts: Counter[str] = Counter()

        # Cashtag form bypasses the blacklist — explicit intent.
        for match in _CASHTAG_RE.finditer(text):
            symbol = match.group(1).upper()
            if symbol in self._valid:
                counts[symbol] += 1

        # Contextual form goes through the blacklist.
        for match in _CONTEXTUAL_RE.finditer(text):
            symbol = match.group(1).upper()
            if symbol in self._blacklist:
                continue
            if symbol in self._valid:
                counts[symbol] += 1

        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def load_alpaca_active_equities(settings: Settings) -> set[str]:
    """Return the set of active Alpaca US equity symbols, cached on disk for 24h.

    The cache lives at ``<aitrade_data_dir>/alpaca_assets.json``. If the file is
    fresh (mtime within TTL), it is read directly; otherwise the function calls
    Alpaca's ``get_all_assets`` and rewrites the cache.

    Heavy dependencies (the Alpaca SDK) are imported lazily so this module can
    be imported cheaply by tests and the CLI.
    """
    cache_path = settings.aitrade_data_dir / _ASSET_CACHE_FILE

    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < _ASSET_CACHE_TTL_SECS:
            try:
                data = json.loads(cache_path.read_text())
                if isinstance(data, list):
                    logger.debug("alpaca_assets cache hit (age={:.0f}s)", age)
                    return {str(s).upper() for s in data}
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("alpaca_assets cache unreadable, refetching: {}", e)

    # Lazy import — alpaca-py is a heavy dep we don't want at import time.
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    client = TradingClient(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=True,
    )
    request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
    assets = client.get_all_assets(request)

    symbols: set[str] = set()
    for asset in assets:
        symbol = getattr(asset, "symbol", None)
        if symbol:
            symbols.add(str(symbol).upper())

    # Persist for the next caller.
    try:
        settings.aitrade_data_dir.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(sorted(symbols)))
        logger.debug("alpaca_assets cache wrote {} symbols", len(symbols))
    except OSError as e:
        logger.warning("could not persist alpaca_assets cache: {}", e)

    return symbols
