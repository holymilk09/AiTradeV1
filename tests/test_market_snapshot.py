"""MarketSnapshotFetcher and regime classification."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aitrade.data.models import Timeframe
from aitrade.market.freshness import StaleDataError
from aitrade.market.regime import Regime, classify_regime
from aitrade.market.snapshot import (
    MarketSnapshot,
    MarketSnapshotFetcher,
    _detect_session,
    is_in_volatile_open_close,
)

# --------------------------------------------------------------------------- #
# Regime classifier — pure 2x2 grid                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("spy_change_pct", "vix", "expected"),
    [
        (0.5, 15.0, Regime.RISK_ON_LOW_VOL),
        (0.5, 25.0, Regime.RISK_ON_HIGH_VOL),
        (-0.5, 15.0, Regime.RISK_OFF_LOW_VOL),
        (-0.5, 25.0, Regime.RISK_OFF_HIGH_VOL),
        # Boundary: spy=0 is risk-on, vix=20 is high-vol
        (0.0, 20.0, Regime.RISK_ON_HIGH_VOL),
    ],
)
def test_classify_regime(spy_change_pct: float, vix: float, expected: Regime) -> None:
    assert classify_regime(spy_change_pct, vix) is expected


# --------------------------------------------------------------------------- #
# Session detection                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("ny_dt", "expected"),
    [
        # Tuesday 2026-04-21
        (datetime(2026, 4, 21, 5, 0), "premarket"),
        (datetime(2026, 4, 21, 9, 0), "premarket"),
        (datetime(2026, 4, 21, 9, 30), "regular"),
        (datetime(2026, 4, 21, 12, 0), "regular"),
        (datetime(2026, 4, 21, 15, 59), "regular"),
        (datetime(2026, 4, 21, 16, 0), "afterhours"),
        (datetime(2026, 4, 21, 19, 59), "afterhours"),
        (datetime(2026, 4, 21, 20, 0), "closed"),
        (datetime(2026, 4, 21, 3, 0), "closed"),
        # Saturday 2026-04-25 — always closed
        (datetime(2026, 4, 25, 12, 0), "closed"),
        # Sunday 2026-04-26 — always closed
        (datetime(2026, 4, 26, 12, 0), "closed"),
    ],
)
def test_detect_session(ny_dt: datetime, expected: str) -> None:
    from zoneinfo import ZoneInfo

    aware = ny_dt.replace(tzinfo=ZoneInfo("America/New_York"))
    assert _detect_session(aware.astimezone(UTC)) == expected


# --------------------------------------------------------------------------- #
# Fetcher: caching, TTL, stale fallback                                        #
# --------------------------------------------------------------------------- #


def _df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-04-15", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def _wire_mock(
    mock: MagicMock,
    *,
    vix_close: float | None = 19.0,
    sector_pcts: dict[str, float] | None = None,
) -> None:
    """Set up the mock data client to return canned daily bars per symbol.

    SPY/QQQ/VIXY have fixed canned series. Sector ETFs return per-symbol
    canned series when the test passes ``sector_pcts``; otherwise sector
    fetches return an empty frame (treated as a per-ETF best-effort miss
    by the fetcher).
    """

    sector_pcts = sector_pcts or {}

    def side_effect(
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        **_: Any,
    ) -> pd.DataFrame:
        if symbol == "SPY":
            return _df([500.0, 505.0])  # +1.0% change
        if symbol == "QQQ":
            return _df([400.0, 396.0])  # -1.0% change
        if symbol == "VIXY":
            if vix_close is None:
                return pd.DataFrame()
            return _df([vix_close - 1, vix_close])
        if symbol in sector_pcts:
            base = 100.0
            return _df([base, base * (1.0 + sector_pcts[symbol] / 100.0)])
        # Unknown sector ETF — return empty (per-ETF best-effort miss).
        return pd.DataFrame()

    mock.fetch_stock_bars.side_effect = side_effect


def test_fetch_returns_snapshot_with_regime() -> None:
    data = MagicMock()
    _wire_mock(data, vix_close=15.0)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=30, max_stale_secs=300)

    snap = fetcher.fetch()

    assert isinstance(snap, MarketSnapshot)
    assert snap.spy_price == 505.0
    assert snap.spy_change_pct == pytest.approx(1.0)
    assert snap.qqq_price == 396.0
    assert snap.qqq_change_pct == pytest.approx(-1.0)
    assert snap.vix == 15.0
    # SPY +1% → risk_on; VIX 15 → low_vol
    assert snap.regime is Regime.RISK_ON_LOW_VOL


def test_fetch_uses_default_vix_when_proxy_unavailable() -> None:
    data = MagicMock()
    _wire_mock(data, vix_close=None)
    fetcher = MarketSnapshotFetcher(data=data)
    snap = fetcher.fetch()
    assert snap.vix == 18.0


def test_fetch_includes_sector_etf_changes() -> None:
    """Phase 5: SPDR sector ETF pct changes ride along on the snapshot."""
    data = MagicMock()
    _wire_mock(
        data,
        sector_pcts={"XLK": 1.2, "XLF": -0.4, "XLE": 0.0},
    )
    fetcher = MarketSnapshotFetcher(data=data)

    snap = fetcher.fetch()

    # Sectors that returned canned bars are present with their pct change;
    # sectors that returned empty frames are dropped (best-effort per ETF).
    assert snap.sector_changes_pct["XLK"] == pytest.approx(1.2)
    assert snap.sector_changes_pct["XLF"] == pytest.approx(-0.4)
    assert snap.sector_changes_pct["XLE"] == pytest.approx(0.0)
    # Unwired sectors must not be present.
    assert "XLU" not in snap.sector_changes_pct


def test_fetch_tolerates_total_sector_failure() -> None:
    """A failure to fetch *every* sector still yields a valid snapshot."""
    data = MagicMock()
    _wire_mock(data)  # no sector_pcts → all sectors return empty
    fetcher = MarketSnapshotFetcher(data=data)
    snap = fetcher.fetch()
    assert snap.sector_changes_pct == {}
    # Other fields still populated normally.
    assert snap.spy_change_pct == pytest.approx(1.0)


def test_cache_hit_within_ttl() -> None:
    data = MagicMock()
    _wire_mock(data)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=30)

    first = fetcher.fetch()
    second = fetcher.fetch()
    assert first is second
    # First fetch hits SPY + QQQ + VIXY + 11 sector ETFs = 14 calls.
    # Cache hit on the second fetch leaves call_count untouched.
    initial_calls = data.fetch_stock_bars.call_count
    assert initial_calls > 3  # picked up sector ETFs too
    # Second fetch was a pure cache hit — no new calls.
    fetcher.fetch()
    assert data.fetch_stock_bars.call_count == initial_calls


def test_cache_miss_after_ttl_elapses() -> None:
    data = MagicMock()
    _wire_mock(data)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=30)

    first = fetcher.fetch()
    initial_calls = data.fetch_stock_bars.call_count
    # Backdate the cached snapshot to look TTL-stale.
    object.__setattr__(
        first, "fetched_at", first.fetched_at - timedelta(seconds=120)
    )
    second = fetcher.fetch()

    assert second is not first
    # Second fetch must repeat the full set of underlying calls.
    assert data.fetch_stock_bars.call_count == initial_calls * 2


def test_force_refresh_bypasses_cache() -> None:
    data = MagicMock()
    _wire_mock(data)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=300)

    fetcher.fetch()
    initial_calls = data.fetch_stock_bars.call_count
    fetcher.fetch(force_refresh=True)
    # Forced refetch repeats the full set.
    assert data.fetch_stock_bars.call_count == initial_calls * 2


def test_stale_fallback_within_max_stale_secs() -> None:
    data = MagicMock()
    _wire_mock(data)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=10, max_stale_secs=600)

    first = fetcher.fetch()
    # Make cache TTL-stale but inside the max-stale budget.
    object.__setattr__(
        first, "fetched_at", first.fetched_at - timedelta(seconds=120)
    )

    # Next fetch fails — but a primary-symbol failure must surface, while a
    # cached snapshot is still acceptable.
    data.fetch_stock_bars.side_effect = RuntimeError("network down")
    second = fetcher.fetch()
    assert second is first


def test_stale_data_error_when_cache_too_old() -> None:
    data = MagicMock()
    _wire_mock(data)
    fetcher = MarketSnapshotFetcher(data=data, ttl_secs=10, max_stale_secs=60)

    first = fetcher.fetch()
    # Backdate cache beyond max_stale_secs.
    object.__setattr__(
        first, "fetched_at", first.fetched_at - timedelta(seconds=600)
    )
    data.fetch_stock_bars.side_effect = RuntimeError("network down")

    with pytest.raises(StaleDataError):
        fetcher.fetch()


def test_age_secs_uses_provided_clock() -> None:
    snap = MarketSnapshot(
        spy_price=1.0,
        spy_change_pct=0.0,
        qqq_price=1.0,
        qqq_change_pct=0.0,
        vix=18.0,
        regime=Regime.RISK_ON_LOW_VOL,
        session="regular",
        fetched_at=datetime(2026, 4, 24, 12, 0, tzinfo=UTC),
    )
    later = datetime(2026, 4, 24, 12, 1, tzinfo=UTC)
    assert snap.age_secs(later) == 60.0


# --------------------------------------------------------------------------- #
# Phase 6 — time-of-day gating                                                #
# --------------------------------------------------------------------------- #


def _ny_aware(*ymdhms: int) -> datetime:
    """Build a NY-time-aware datetime from (Y, M, D, H, M[, S])."""
    from zoneinfo import ZoneInfo

    return datetime(*ymdhms, tzinfo=ZoneInfo("America/New_York"))  # type: ignore[arg-type]


def test_open_window_gates_first_5_minutes() -> None:
    """9:30 → 9:35 ET on a weekday is gated by default."""
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 9, 30)) is True
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 9, 34, 59)) is True
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 9, 35)) is False  # boundary


def test_close_window_gates_last_5_minutes() -> None:
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 15, 55)) is True
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 15, 59, 59)) is True
    # 16:00 sharp is the close itself — outside the *open* regular session,
    # so the function returns False (the engine doesn't trade then anyway).
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 15, 54, 59)) is False


def test_midday_is_not_gated() -> None:
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 12, 0)) is False


def test_zero_skip_disables_gating() -> None:
    assert is_in_volatile_open_close(
        _ny_aware(2026, 4, 21, 9, 32),
        skip_open_mins=0, skip_close_mins=0,
    ) is False


def test_weekend_never_gated() -> None:
    # Saturday at what would be the open — no regular session exists.
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 25, 9, 32)) is False


def test_custom_skip_window_extends_gate() -> None:
    """skip_open_mins=15 widens the gate to 9:30 → 9:45."""
    assert is_in_volatile_open_close(
        _ny_aware(2026, 4, 21, 9, 44),
        skip_open_mins=15,
    ) is True
    assert is_in_volatile_open_close(
        _ny_aware(2026, 4, 21, 9, 46),
        skip_open_mins=15,
    ) is False


def test_premarket_not_gated() -> None:
    """The gate only applies *inside* regular session, not premarket."""
    assert is_in_volatile_open_close(_ny_aware(2026, 4, 21, 8, 0)) is False
