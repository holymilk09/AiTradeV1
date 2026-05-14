"""Market-wide snapshot of SPY/QQQ/VIX with freshness + caching.

The :class:`MarketSnapshotFetcher` is the single entry point strategies use
to read "what is the market doing right now". It enforces:

- a short in-memory TTL so we don't hammer the data feed;
- a max-stale budget so a transient fetch failure doesn't poison decisions;
- explicit session detection in NY-time (premarket/regular/afterhours/closed);
- a regime label derived from SPY's 1-day change and the latest VIX.

VIX caveat
----------
VIX itself (``^VIX``) is not generally exposed via Alpaca's stock historical
data feed. We use ``VIXY`` (the ProShares VIX Short-Term Futures ETF) as a
proxy. If even that fails we fall back to a neutral default of ``18.0`` so
strategies still get *some* regime signal — but the fetcher logs a warning
so the operator notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from loguru import logger

from aitrade.market.regime import Regime, classify_regime

if TYPE_CHECKING:
    from aitrade.data.alpaca_data import AlpacaDataClient

_NY = ZoneInfo("America/New_York")
_DEFAULT_VIX = 18.0
_VIX_PROXY_SYMBOL = "VIXY"

# SPDR sector ETFs — the standard "what's leading the market" basket.
# Keys are the ETF symbol; values are the human-readable sector label so the
# floor-trader prompt can interpret them without a translation step.
_SECTOR_ETFS: dict[str, str] = {
    "XLK": "technology",
    "XLF": "financials",
    "XLE": "energy",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer_discretionary",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLB": "materials",
    "XLRE": "real_estate",
    "XLC": "communications",
}


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Point-in-time read of the broad market.

    All prices and pct-changes use the latest two daily closes available at
    fetch time. ``session`` is computed from ``fetched_at`` translated to
    ``America/New_York``.

    ``sector_changes_pct`` maps each SPDR sector ETF symbol (XLK, XLF, ...)
    to its 1-day pct change. A breakout in NVDA confirmed by XLK leading is
    very different from one fading into a flat tech sector — surfacing this
    to the floor-trader lets it weight sector rotation explicitly.
    """

    spy_price: float
    spy_change_pct: float
    qqq_price: float
    qqq_change_pct: float
    vix: float
    regime: Regime
    session: str
    fetched_at: datetime
    sector_changes_pct: dict[str, float] = field(default_factory=dict)

    def age_secs(self, now: datetime | None = None) -> float:
        """Return how many seconds ago this snapshot was fetched."""

        current = now if now is not None else datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        fetched = self.fetched_at
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=UTC)
        return (current - fetched).total_seconds()


def _detect_session(at: datetime) -> str:
    """Map a UTC instant to a US-equity session string.

    Returns one of ``"premarket"``, ``"regular"``, ``"afterhours"``,
    ``"closed"``. Weekends are always ``"closed"``.
    """

    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    ny = at.astimezone(_NY)
    if ny.weekday() >= 5:  # Saturday=5, Sunday=6
        return "closed"
    t = ny.time()
    if time(4, 0) <= t < time(9, 30):
        return "premarket"
    if time(9, 30) <= t < time(16, 0):
        return "regular"
    if time(16, 0) <= t < time(20, 0):
        return "afterhours"
    return "closed"


def is_in_volatile_open_close(
    at: datetime,
    *,
    skip_open_mins: int = 5,
    skip_close_mins: int = 5,
) -> bool:
    """Phase 6: True when the instant lands in the regular session's first
    ``skip_open_mins`` minutes (9:30 ET → 9:30+N) or the last ``skip_close_mins``
    (16:00−N → 16:00 ET).

    Returns False outside regular session, on weekends, or when both knobs
    are 0. Used by the engine to skip cycles in chronically noisy windows
    that historically just generate stop-outs.
    """
    if skip_open_mins <= 0 and skip_close_mins <= 0:
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    ny = at.astimezone(_NY)
    if ny.weekday() >= 5:
        return False
    t = ny.time()
    open_end = time(9, 30 + skip_open_mins) if skip_open_mins > 0 else time(9, 30)
    close_start = (
        time(15, 60 - skip_close_mins) if skip_close_mins > 0 else time(16, 0)
    )
    if skip_open_mins > 0 and time(9, 30) <= t < open_end:
        return True
    return skip_close_mins > 0 and close_start <= t < time(16, 0)


@dataclass(slots=True)
class MarketSnapshotFetcher:
    """TTL-cached fetcher for the broad-market snapshot.

    Parameters
    ----------
    data:
        Configured :class:`AlpacaDataClient`. Used to pull daily bars for
        SPY, QQQ, and the VIX proxy.
    ttl_secs:
        How long a successful snapshot remains "fresh"; calls within this
        window return the cached snapshot without hitting the data feed.
    max_stale_secs:
        How old the cache may be when serving as a fallback after a fetch
        failure. Beyond this, a :class:`StaleDataError` is raised.
    """

    data: AlpacaDataClient
    ttl_secs: int = 30
    max_stale_secs: int = 300
    _cached: MarketSnapshot | None = field(default=None, init=False, repr=False)

    def fetch(self, force_refresh: bool = False) -> MarketSnapshot:
        """Return the latest market snapshot, refetching if stale.

        On fetch failure, falls back to the cached snapshot if it is still
        within ``max_stale_secs``; otherwise raises :class:`StaleDataError`.
        """

        # Lazy import to avoid a circular dep with the freshness module on
        # package init (and to keep heavy imports off the hot path).
        from aitrade.market.freshness import StaleDataError

        now = datetime.now(UTC)
        if (
            not force_refresh
            and self._cached is not None
            and self._cached.age_secs(now) <= self.ttl_secs
        ):
            return self._cached

        try:
            snapshot = self._fetch_fresh(now)
        except Exception as exc:  # noqa: BLE001 — explicit fallback path
            if (
                self._cached is not None
                and self._cached.age_secs(now) <= self.max_stale_secs
            ):
                logger.warning(
                    "market snapshot refetch failed ({}); serving cached "
                    "snapshot age={:.1f}s",
                    exc,
                    self._cached.age_secs(now),
                )
                return self._cached
            raise StaleDataError(
                f"market snapshot refetch failed and no fresh cache: {exc}"
            ) from exc

        self._cached = snapshot
        return snapshot

    def _fetch_fresh(self, now: datetime) -> MarketSnapshot:
        """Pull SPY/QQQ/VIX-proxy + sector ETFs and assemble the snapshot."""

        start = now - timedelta(days=10)
        spy_price, spy_change = self._latest_two_close_change("SPY", start, now)
        qqq_price, qqq_change = self._latest_two_close_change("QQQ", start, now)
        vix = self._fetch_vix(start, now)
        sectors = self._fetch_sectors(start, now)

        return MarketSnapshot(
            spy_price=spy_price,
            spy_change_pct=spy_change,
            qqq_price=qqq_price,
            qqq_change_pct=qqq_change,
            vix=vix,
            regime=classify_regime(spy_change, vix),
            session=_detect_session(now),
            fetched_at=now,
            sector_changes_pct=sectors,
        )

    def _fetch_sectors(
        self, start: datetime, end: datetime
    ) -> dict[str, float]:
        """Best-effort sector ETF reads. Per-ETF failure drops just that key.

        Returns a partial dict on partial failure — sector breadth is useful
        even if one ETF couldn't be read. Empty dict only on total failure
        (which the cycle still proceeds with; floor-trader sees no sector
        info this cycle).
        """
        out: dict[str, float] = {}
        for symbol in _SECTOR_ETFS:
            try:
                _, pct = self._latest_two_close_change(symbol, start, end)
            except Exception as exc:  # noqa: BLE001 — per-ETF best-effort
                logger.debug(
                    "sector ETF {} unavailable ({}); skipping", symbol, exc
                )
                continue
            out[symbol] = round(pct, 3)
        return out

    def _latest_two_close_change(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> tuple[float, float]:
        """Return ``(latest_close, pct_change_vs_prior_close)`` for ``symbol``."""

        from aitrade.data.models import Timeframe

        df = self.data.fetch_stock_bars(symbol, Timeframe.DAY_1, start, end)
        if df is None or len(df) == 0:
            raise RuntimeError(f"no daily bars returned for {symbol}")
        closes = list(df["close"].astype(float))
        if len(closes) < 2:
            raise RuntimeError(
                f"need >=2 daily closes for {symbol}, got {len(closes)}"
            )
        latest = closes[-1]
        prior = closes[-2]
        pct = (latest - prior) / prior * 100.0 if prior else 0.0
        return latest, pct

    def _fetch_vix(self, start: datetime, end: datetime) -> float:
        """Best-effort VIX read via the ``VIXY`` proxy with default fallback."""

        from aitrade.data.models import Timeframe

        try:
            df = self.data.fetch_stock_bars(
                _VIX_PROXY_SYMBOL, Timeframe.DAY_1, start, end
            )
            if df is None or len(df) == 0:
                raise RuntimeError("empty bars")
            return float(df["close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001 — proxy is best-effort
            logger.warning(
                "VIX proxy {} unavailable ({}); falling back to default {}",
                _VIX_PROXY_SYMBOL,
                exc,
                _DEFAULT_VIX,
            )
            return _DEFAULT_VIX
