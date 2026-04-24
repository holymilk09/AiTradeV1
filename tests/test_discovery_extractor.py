"""Tests for the discovery TickerExtractor + Alpaca asset cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.discovery.extractor import TickerExtractor, load_alpaca_active_equities


def test_cashtag_extraction_finds_symbols() -> None:
    extractor = TickerExtractor(valid_tickers={"AAPL", "TSLA", "NVDA"})
    text = "Watching $AAPL and $TSLA today, also keeping an eye on $NVDA."
    result = extractor.extract(text)
    symbols = {sym for sym, _ in result}
    assert symbols == {"AAPL", "TSLA", "NVDA"}


def test_cashtag_counts_mentions() -> None:
    extractor = TickerExtractor(valid_tickers={"AAPL"})
    text = "$AAPL $AAPL $AAPL is everywhere"
    result = extractor.extract(text)
    assert result == [("AAPL", 3)]


def test_contextual_extraction_finds_symbols() -> None:
    extractor = TickerExtractor(valid_tickers={"NVDA", "AMD"})
    text = "NVDA stock is on a rally; AMD shares jumped too."
    result = extractor.extract(text)
    symbols = {sym for sym, _ in result}
    assert symbols == {"NVDA", "AMD"}


def test_contextual_blacklist_blocks_common_words() -> None:
    """`OR puts` is contextual; OR is in the default blacklist → must drop."""
    extractor = TickerExtractor(valid_tickers={"OR"})
    text = "OR puts are flying"
    result = extractor.extract(text)
    assert result == []


def test_cashtag_bypasses_blacklist() -> None:
    """`$IT` is explicit market intent — blacklist must not apply."""
    extractor = TickerExtractor(valid_tickers={"IT"})
    text = "$IT is a real ticker that's surging"
    result = extractor.extract(text)
    assert result == [("IT", 1)]


def test_invalid_ticker_filtered_out() -> None:
    """Cashtags that aren't in the universe must be dropped."""
    extractor = TickerExtractor(valid_tickers={"AAPL"})
    text = "$AAPL and $FAKEFAKE both moving"
    result = extractor.extract(text)
    assert result == [("AAPL", 1)]


def test_results_sorted_by_count_desc() -> None:
    extractor = TickerExtractor(valid_tickers={"AAPL", "TSLA", "NVDA"})
    text = "$NVDA $NVDA $NVDA $AAPL $AAPL $TSLA"
    result = extractor.extract(text)
    assert result == [("NVDA", 3), ("AAPL", 2), ("TSLA", 1)]


def test_custom_blacklist_overrides_default() -> None:
    """Passing a custom blacklist replaces, not augments, the default."""
    extractor = TickerExtractor(valid_tickers={"OR", "FOO"}, blacklist={"FOO"})
    text = "OR shares jumped; FOO calls active"
    result = extractor.extract(text)
    symbols = {sym for sym, _ in result}
    assert "OR" in symbols  # no longer blacklisted
    assert "FOO" not in symbols  # newly blacklisted


def test_cashtag_and_contextual_combine() -> None:
    extractor = TickerExtractor(valid_tickers={"AAPL"})
    text = "$AAPL is up and AAPL stock keeps rallying"
    result = extractor.extract(text)
    assert result == [("AAPL", 2)]


def test_load_alpaca_uses_disk_cache_when_fresh(tmp_path: Path) -> None:
    """Fresh on-disk cache must short-circuit the Alpaca call entirely."""
    settings = Settings(
        alpaca_api_key=SecretStr("test"),
        alpaca_secret_key=SecretStr("test"),
        aitrade_data_dir=tmp_path,
        aitrade_log_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()

    cache_path = tmp_path / "alpaca_assets.json"
    cache_path.write_text(json.dumps(["AAPL", "TSLA", "NVDA"]))

    # Sentinel: if alpaca_active_equities reaches the SDK, this would import-fail
    # in CI. Patch TradingClient to ensure it is never instantiated.
    with patch("alpaca.trading.client.TradingClient") as mock_client:
        result = load_alpaca_active_equities(settings)

    assert result == {"AAPL", "TSLA", "NVDA"}
    mock_client.assert_not_called()


def test_load_alpaca_refetches_when_cache_stale(tmp_path: Path) -> None:
    """A cache file older than the TTL must trigger a refetch."""
    settings = Settings(
        alpaca_api_key=SecretStr("test"),
        alpaca_secret_key=SecretStr("test"),
        aitrade_data_dir=tmp_path,
        aitrade_log_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()

    cache_path = tmp_path / "alpaca_assets.json"
    cache_path.write_text(json.dumps(["STALE"]))
    # Backdate well past the 24h TTL.
    old_time = time.time() - (48 * 60 * 60)
    import os

    os.utime(cache_path, (old_time, old_time))

    fake_asset = type("A", (), {"symbol": "FRESH"})()

    with patch("alpaca.trading.client.TradingClient") as mock_client_cls:
        instance = mock_client_cls.return_value
        instance.get_all_assets.return_value = [fake_asset]
        result = load_alpaca_active_equities(settings)

    assert "FRESH" in result
    mock_client_cls.assert_called_once()
