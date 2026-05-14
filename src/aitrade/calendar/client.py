"""Financial Modeling Prep economic + earnings calendar client.

Free tier: 250 requests/day — far more than we need for one daily refresh.
Both endpoints return small JSON arrays so streaming is unnecessary.

Design notes
------------
* All public methods catch every exception and return ``[]``. The calendar
  is an LLM-prompt enrichment, never an order gate. A flaky upstream must
  never block the engine.
* Caching is process-local with a 4-hour TTL by default; macro events don't
  shift more than once a day, and earnings dates are stable for the week.
* Tests inject a stub HTTP client by setting ``client._http`` directly. The
  stub only needs ``get(url, params=...) -> object`` where the returned
  object exposes ``.status_code`` and ``.json()``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import httpx
from loguru import logger

_BASE_URL = "https://financialmodelingprep.com/api/v3"
_TIMEOUT_SECS = 15.0
_RETRY_BACKOFFS_SECS = (1.0, 2.0)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _parse_dt(raw: object) -> datetime | None:
    """Parse FMP date strings ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'). UTC-naive→UTC."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    return None


def _maybe_float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(cast(float, raw))
    except (TypeError, ValueError):
        return None


def _maybe_str(raw: object) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    return str(raw)


@dataclass(frozen=True, slots=True)
class EconomicEvent:
    """One row from FMP's ``/economic_calendar``."""

    date: datetime
    event: str
    country: str
    currency: str | None
    previous: float | None
    estimate: float | None
    actual: float | None
    change_pct: float | None
    impact: str | None

    def to_compact(self) -> dict[str, str | float | None]:
        """Prompt-friendly projection. Numeric fields rounded to 4 dp."""
        return {
            "date": self.date.strftime("%Y-%m-%d %H:%M"),
            "event": self.event,
            "prev": _round_or_none(self.previous),
            "est": _round_or_none(self.estimate),
            "actual": _round_or_none(self.actual),
            "impact": self.impact,
        }


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    """One row from FMP's ``/earning_calendar``."""

    date: datetime
    symbol: str
    time: str | None
    eps_estimated: float | None
    eps_actual: float | None
    revenue_estimated: float | None
    revenue_actual: float | None

    def to_compact(self) -> dict[str, str | float | None]:
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "symbol": self.symbol,
            "time": self.time,
            "eps_est": _round_or_none(self.eps_estimated, 4),
            "eps_actual": _round_or_none(self.eps_actual, 4),
            "rev_est": _round_or_none(self.revenue_estimated, 0),
            "rev_actual": _round_or_none(self.revenue_actual, 0),
        }


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    # Heterogeneous per-key payload; the caller knows the concrete type because
    # cache keys are namespaced ("econ::..." vs "earn::...").
    value: list[EconomicEvent] | list[EarningsEvent]


class EconomicCalendarClient:
    """Cached client over FMP's free-tier economic + earnings calendars."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        ttl_secs: int = 14_400,
        http: object | None = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("FMP_API_KEY", "").strip() or None
        self._api_key: str | None = api_key
        self._ttl_secs: int = int(ttl_secs)
        # ``object`` so tests can inject any duck-typed stub with .get().
        self._http: object | None = http
        self._cache: dict[str, _CacheEntry] = {}

    # -- HTTP plumbing -----------------------------------------------------

    def _get_http(self) -> object:
        if self._http is None:
            self._http = httpx.Client(timeout=_TIMEOUT_SECS)
        return self._http

    def _request(self, path: str, params: dict[str, str]) -> list[object] | None:
        """GET ``path`` with retries. Returns parsed list or None on failure."""
        if self._api_key is None:
            logger.warning("FMP_API_KEY missing; calendar request skipped ({}).", path)
            return None
        url = f"{_BASE_URL}{path}"
        full_params = {**params, "apikey": self._api_key}
        http = self._get_http()
        last_err: Exception | None = None
        # 1 initial attempt + len(_RETRY_BACKOFFS_SECS) retries.
        for attempt in range(1 + len(_RETRY_BACKOFFS_SECS)):
            try:
                resp = http.get(url, params=full_params)  # type: ignore[attr-defined]
            except (httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                last_err = exc
                if attempt < len(_RETRY_BACKOFFS_SECS):
                    time.sleep(_RETRY_BACKOFFS_SECS[attempt])
                    continue
                logger.warning("FMP {} transport error after retries: {}", path, exc)
                return None
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("FMP {} unexpected error: {}", path, exc)
                return None

            status = getattr(resp, "status_code", 0)
            if status != 200:
                logger.warning("FMP {} returned status {}", path, status)
                return None
            try:
                payload = resp.json()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("FMP {} JSON decode failed: {}", path, exc)
                return None
            if not isinstance(payload, list):
                logger.warning("FMP {} unexpected payload type: {}", path, type(payload))
                return None
            # We deliberately keep the inner element type as ``object`` so the
            # parsing loop's ``isinstance(row, dict)`` guard is meaningful to
            # the type-checker and to humans reading the code.
            return cast(list[object], payload)
        # Loop exits only via return; this is unreachable but mypy needs it.
        if last_err is not None:  # pragma: no cover
            logger.warning("FMP {} exhausted retries: {}", path, last_err)
        return None

    # -- Cache -------------------------------------------------------------

    def _cache_get_econ(self, key: str) -> list[EconomicEvent] | None:
        entry = self._cache.get(key)
        if entry is None or entry.expires_at < time.monotonic():
            return None
        value = entry.value
        # Type narrowing for mypy strict.
        if value and isinstance(value[0], EarningsEvent):
            return None
        return cast(list[EconomicEvent], value)

    def _cache_get_earn(self, key: str) -> list[EarningsEvent] | None:
        entry = self._cache.get(key)
        if entry is None or entry.expires_at < time.monotonic():
            return None
        value = entry.value
        if value and isinstance(value[0], EconomicEvent):
            return None
        return cast(list[EarningsEvent], value)

    def _cache_put(
        self, key: str, value: list[EconomicEvent] | list[EarningsEvent]
    ) -> None:
        self._cache[key] = _CacheEntry(
            expires_at=time.monotonic() + self._ttl_secs,
            value=value,
        )

    # -- Public API --------------------------------------------------------

    def economic_events(
        self,
        *,
        days_ahead: int = 7,
        days_back: int = 0,
        us_only: bool = True,
    ) -> list[EconomicEvent]:
        """Fetch upcoming macro events. Returns ``[]`` on any failure."""
        cache_key = f"econ::{days_ahead}::{days_back}::{us_only}"
        cached = self._cache_get_econ(cache_key)
        if cached is not None:
            return cached

        try:
            today = datetime.now(UTC).date()
            params = {
                "from": (today - timedelta(days=max(0, days_back))).isoformat(),
                "to": (today + timedelta(days=max(0, days_ahead))).isoformat(),
            }
            rows = self._request("/economic_calendar", params)
            if rows is None:
                self._cache_put(cache_key, [])
                return []

            out: list[EconomicEvent] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                country = _maybe_str(row.get("country")) or ""
                if us_only and country.lower() != "us":
                    continue
                dt = _parse_dt(row.get("date"))
                if dt is None:
                    continue
                event_name = _maybe_str(row.get("event"))
                if event_name is None:
                    continue
                out.append(
                    EconomicEvent(
                        date=dt,
                        event=event_name,
                        country=country,
                        currency=_maybe_str(row.get("currency")),
                        previous=_maybe_float(row.get("previous")),
                        estimate=_maybe_float(row.get("estimate")),
                        actual=_maybe_float(row.get("actual")),
                        change_pct=_maybe_float(row.get("changePercentage")),
                        impact=_maybe_str(row.get("impact")),
                    )
                )
            out.sort(key=lambda e: e.date)
            self._cache_put(cache_key, out)
            return out
        except Exception as exc:
            logger.warning("economic_events failed: {}", exc)
            return []

    def earnings_events(
        self,
        *,
        days_ahead: int = 7,
        symbols: list[str] | None = None,
    ) -> list[EarningsEvent]:
        """Fetch upcoming earnings. Returns ``[]`` on any failure."""
        sym_set: set[str] | None = None
        if symbols is not None:
            sym_set = {s.upper() for s in symbols if s}

        # Cache key includes the sorted symbol filter so different filters
        # don't collide.
        sym_key = ",".join(sorted(sym_set)) if sym_set is not None else "*"
        cache_key = f"earn::{days_ahead}::{sym_key}"
        cached = self._cache_get_earn(cache_key)
        if cached is not None:
            return cached

        try:
            today = datetime.now(UTC).date()
            params = {
                "from": today.isoformat(),
                "to": (today + timedelta(days=max(0, days_ahead))).isoformat(),
            }
            rows = self._request("/earning_calendar", params)
            if rows is None:
                self._cache_put(cache_key, [])
                return []

            out: list[EarningsEvent] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = _maybe_str(row.get("symbol"))
                if symbol is None:
                    continue
                symbol = symbol.upper()
                if sym_set is not None and symbol not in sym_set:
                    continue
                dt = _parse_dt(row.get("date"))
                if dt is None:
                    continue
                out.append(
                    EarningsEvent(
                        date=dt,
                        symbol=symbol,
                        time=_maybe_str(row.get("time")),
                        eps_estimated=_maybe_float(row.get("epsEstimated")),
                        eps_actual=_maybe_float(row.get("eps")),
                        revenue_estimated=_maybe_float(row.get("revenueEstimated")),
                        revenue_actual=_maybe_float(row.get("revenue")),
                    )
                )
            out.sort(key=lambda e: (e.date, e.symbol))
            self._cache_put(cache_key, out)
            return out
        except Exception as exc:
            logger.warning("earnings_events failed: {}", exc)
            return []
