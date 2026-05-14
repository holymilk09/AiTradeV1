"""Tests for ``aitrade.calendar``.

The HTTP layer is exercised via a duck-typed stub injected on ``client._http``.
We avoid module-level monkeypatching so unrelated import-order shifts don't
brittle these tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from aitrade.calendar import EconomicCalendarClient


class _StubResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class _StubHTTP:
    """Records every call and returns a queued response per request."""

    def __init__(self, responses: list[_StubResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str] | None = None) -> _StubResponse:
        self.calls.append((url, dict(params or {})))
        if not self.responses:
            raise AssertionError("no more stub responses queued")
        return self.responses.pop(0)


@pytest.fixture
def econ_payload() -> list[dict[str, Any]]:
    # Out-of-order on purpose; client must sort ascending.
    return [
        {
            "date": "2026-05-10 12:30:00",
            "event": "CPI YoY",
            "country": "US",
            "currency": "USD",
            "previous": 3.2,
            "estimate": 3.1,
            "actual": None,
            "changePercentage": -0.1,
            "impact": "High",
        },
        {
            "date": "2026-05-05 14:00:00",
            "event": "FOMC Statement",
            "country": "us",  # case-insensitive
            "currency": "USD",
            "previous": None,
            "estimate": None,
            "actual": None,
            "changePercentage": None,
            "impact": "High",
        },
        {
            "date": "not-a-date",  # should be discarded
            "event": "Junk",
            "country": "US",
            "currency": "USD",
            "previous": None,
            "estimate": None,
            "actual": None,
            "changePercentage": None,
            "impact": None,
        },
    ]


def test_economic_events_parses_response(econ_payload: list[dict[str, Any]]) -> None:
    stub = _StubHTTP([_StubResponse(200, econ_payload)])
    client = EconomicCalendarClient(api_key="dummy")
    client._http = stub

    events = client.economic_events(days_ahead=14)

    assert len(events) == 2
    # Sorted ascending — FOMC (May 5) before CPI (May 10).
    assert events[0].event == "FOMC Statement"
    assert events[1].event == "CPI YoY"
    assert events[1].previous == pytest.approx(3.2)
    assert events[1].estimate == pytest.approx(3.1)
    assert events[1].impact == "High"
    # Stub was called exactly once with the right path.
    assert len(stub.calls) == 1
    assert stub.calls[0][0].endswith("/economic_calendar")
    assert stub.calls[0][1]["apikey"] == "dummy"


def test_economic_events_us_only() -> None:
    payload = [
        {
            "date": "2026-05-05 09:00:00",
            "event": "ECB Rate Decision",
            "country": "EU",
            "currency": "EUR",
            "previous": 4.0,
            "estimate": 4.0,
            "actual": None,
            "changePercentage": None,
            "impact": "High",
        },
        {
            "date": "2026-05-06 12:30:00",
            "event": "Nonfarm Payrolls",
            "country": "US",
            "currency": "USD",
            "previous": 200000,
            "estimate": 180000,
            "actual": None,
            "changePercentage": None,
            "impact": "High",
        },
        {
            "date": "2026-05-07 02:00:00",
            "event": "BOJ Statement",
            "country": "JP",
            "currency": "JPY",
            "previous": None,
            "estimate": None,
            "actual": None,
            "changePercentage": None,
            "impact": "Medium",
        },
    ]
    stub = _StubHTTP([_StubResponse(200, payload)])
    client = EconomicCalendarClient(api_key="dummy")
    client._http = stub

    events = client.economic_events(days_ahead=14, us_only=True)
    assert [e.event for e in events] == ["Nonfarm Payrolls"]
    assert events[0].country.lower() == "us"


def test_economic_events_caches_within_ttl(econ_payload: list[dict[str, Any]]) -> None:
    stub = _StubHTTP([_StubResponse(200, econ_payload)])
    client = EconomicCalendarClient(api_key="dummy", ttl_secs=3600)
    client._http = stub

    first = client.economic_events(days_ahead=7)
    second = client.economic_events(days_ahead=7)

    assert first == second
    assert len(stub.calls) == 1, "second call should be served from cache"


def test_economic_events_returns_empty_on_404() -> None:
    stub = _StubHTTP([_StubResponse(404, {"error": "not found"})])
    client = EconomicCalendarClient(api_key="dummy")
    client._http = stub

    events = client.economic_events(days_ahead=7)
    assert events == []
    assert len(stub.calls) == 1


def test_economic_events_returns_empty_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    stub = _StubHTTP([])  # no responses queued — must not be called
    client = EconomicCalendarClient()  # no api_key, no env
    client._http = stub

    events = client.economic_events(days_ahead=7)
    assert events == []
    assert stub.calls == []


def test_earnings_events_filters_by_symbol() -> None:
    payload = [
        {
            "date": "2026-05-08",
            "symbol": "AAPL",
            "eps": None,
            "epsEstimated": 1.5,
            "time": "amc",
            "revenue": None,
            "revenueEstimated": 90_000_000_000,
            "fiscalDateEnding": "2026-03-31",
            "updatedFromDate": "2026-04-30",
        },
        {
            "date": "2026-05-09",
            "symbol": "TSLA",
            "eps": None,
            "epsEstimated": 0.6,
            "time": "amc",
            "revenue": None,
            "revenueEstimated": 25_000_000_000,
            "fiscalDateEnding": "2026-03-31",
            "updatedFromDate": "2026-04-30",
        },
        {
            "date": "2026-05-10",
            "symbol": "MSFT",
            "eps": None,
            "epsEstimated": 2.8,
            "time": "amc",
            "revenue": None,
            "revenueEstimated": 65_000_000_000,
            "fiscalDateEnding": "2026-03-31",
            "updatedFromDate": "2026-04-30",
        },
    ]
    stub = _StubHTTP([_StubResponse(200, payload)])
    client = EconomicCalendarClient(api_key="dummy")
    client._http = stub

    events = client.earnings_events(days_ahead=14, symbols=["aapl", "MSFT"])
    assert sorted(e.symbol for e in events) == ["AAPL", "MSFT"]
    assert len(events) == 2


def test_to_compact_handles_missing_fields() -> None:
    payload = [
        {
            "date": "2026-05-05 14:00:00",
            "event": "FOMC Statement",
            "country": "US",
            "currency": None,
            "previous": None,
            "estimate": None,
            "actual": None,
            "changePercentage": None,
            "impact": None,
        }
    ]
    stub = _StubHTTP([_StubResponse(200, payload)])
    client = EconomicCalendarClient(api_key="dummy")
    client._http = stub

    events = client.economic_events(days_ahead=7)
    assert len(events) == 1
    compact = events[0].to_compact()
    assert compact["event"] == "FOMC Statement"
    assert compact["prev"] is None
    assert compact["est"] is None
    assert compact["actual"] is None
    assert compact["impact"] is None
    # date is always a stringified timestamp
    assert isinstance(compact["date"], str)

    # And earnings to_compact also tolerates Nones.
    earn_payload = [
        {
            "date": "2026-05-08",
            "symbol": "AAPL",
            "eps": None,
            "epsEstimated": None,
            "time": None,
            "revenue": None,
            "revenueEstimated": None,
        }
    ]
    stub2 = _StubHTTP([_StubResponse(200, earn_payload)])
    client2 = EconomicCalendarClient(api_key="dummy")
    client2._http = stub2

    earn = client2.earnings_events(days_ahead=14)
    assert len(earn) == 1
    ec = earn[0].to_compact()
    assert ec["symbol"] == "AAPL"
    assert ec["eps_est"] is None
    assert ec["eps_actual"] is None
    assert ec["rev_est"] is None
    assert ec["time"] is None
