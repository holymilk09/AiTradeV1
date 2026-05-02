"""End-to-end dashboard route tests via FastAPI TestClient.

We do NOT touch Alpaca — the broker is stubbed at app-build time. The
journal is a real SQLite file in tmp_path so the read-only sqlite URI
helpers exercise their normal path.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from aitrade.brokers.base import AccountSummary, BrokerClient, Position
from aitrade.config import Settings
from aitrade.dashboard.app import build_app
from aitrade.execution.orders import OrderAck, OrderRequest, OrderStatus

_USER = "ignored"
_PASS = "tr0ub4dor-correct-horse"


class _StubBroker(BrokerClient):
    """Fake BrokerClient — the dashboard exercises read methods + cancel/submit."""

    def __init__(
        self,
        *,
        positions: list[Position] | None = None,
    ) -> None:
        self._positions = positions or []
        self.cancelled_all = 0
        self.orders_submitted: list[OrderRequest] = []

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> AccountSummary:
        return AccountSummary(
            account_id="stub",
            cash=10_000.0,
            equity=12_345.67,
            buying_power=20_000.0,
            currency="USD",
            is_paper=True,
        )

    def get_positions(self) -> list[Position]:
        return list(self._positions)

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self.orders_submitted.append(order)
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id="stub-1",
            status=OrderStatus.ACCEPTED,
        )

    def cancel_order(self, broker_order_id: str) -> None:  # pragma: no cover
        pass

    def cancel_all(self) -> int:
        self.cancelled_all += 1
        return 0


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        alpaca_api_key=SecretStr("k"),
        alpaca_secret_key=SecretStr("s"),
        anthropic_api_key=SecretStr("ak"),
        aitrade_dashboard_password=SecretStr(_PASS),
        aitrade_data_dir=tmp_path / "data",
        aitrade_log_dir=tmp_path / "logs",
    )


def _seed_journal(log_dir: Path) -> None:
    """Create a minimal SQLite journal with one round-trip + one decision event."""
    log_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(log_dir / "trades.sqlite", isolation_level=None)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trade_events (
            event_id    TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            symbol      TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            payload     TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS round_trips (
            trade_id        TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL,
            strategy_id     TEXT,
            entry_event_id  TEXT NOT NULL,
            exit_event_id   TEXT NOT NULL,
            entry_ts        TEXT NOT NULL,
            exit_ts         TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            exit_price      REAL NOT NULL,
            qty             REAL NOT NULL,
            pnl_usd         REAL NOT NULL,
            pnl_pct         REAL NOT NULL,
            holding_secs    INTEGER NOT NULL,
            pnl_bucket      TEXT NOT NULL,
            exit_reason     TEXT NOT NULL,
            entry_thesis    TEXT,
            entry_catalyst  TEXT,
            entry_confidence REAL,
            pattern_hits    TEXT,
            market_snapshot TEXT
        );
        """
    )
    # Use today's UTC timestamps so the today-card counters tick.
    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    entry_ts = (today_start + timedelta(hours=13, minutes=30)).isoformat()
    exit_ts = (today_start + timedelta(hours=15, minutes=45)).isoformat()
    decision_ts = (today_start + timedelta(hours=13, minutes=25)).isoformat()
    conn.execute(
        "INSERT INTO round_trips VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "rt-1", "AAPL", "engine", "ev-buy", "ev-sell",
            entry_ts, exit_ts,
            172.50, 175.80, 1.0, 3.30, 0.0191, 8100,
            "WIN", "TARGET_HIT", "Volume breakout with SPY bullish",
            "5d high + volume spike", 0.78,
            "[]", "{}",
        ),
    )
    conn.execute(
        "INSERT INTO trade_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "ev-1", "run-1", "engine", "floor_trader_decision", "AAPL",
            decision_ts,
            '{"should_trade": true, "direction": "long", "confidence": 0.78,'
            ' "thesis": "vol breakout", "pick_symbol": "AAPL"}',
        ),
    )
    conn.close()


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, _StubBroker, Settings]:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    _seed_journal(settings.aitrade_log_dir)
    broker = _StubBroker(
        positions=[
            Position(
                symbol="AAPL", qty=2.0, avg_entry_price=170.0,
                market_value=350.0, unrealized_pnl=10.0,
            )
        ],
    )
    app = build_app(settings=settings, broker_factory=lambda: broker)
    # Override the get_settings dependency so the auth dep sees our test password.
    from aitrade.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), broker, settings


def test_unauthenticated_request_returns_401(client) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    r = tc.get("/")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_healthz_is_unauthenticated(client) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    r = tc.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_home_renders_for_authenticated_caller(client) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    r = tc.get("/", auth=(_USER, _PASS))
    assert r.status_code == 200
    body = r.text
    assert "aitrade" in body
    assert "AAPL" in body  # position symbol
    assert "WIN" in body  # round-trip pnl_bucket pill
    assert "+$3.30" in body  # round-trip realized pnl
    assert "vol breakout" in body  # decision thesis
    assert "$12,345.67" in body  # equity formatted


def test_home_rejects_wrong_password(client) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    r = tc.get("/", auth=(_USER, "nope"))
    assert r.status_code == 401


def test_status_api_returns_structured_json(client) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    r = tc.get("/api/status", auth=(_USER, _PASS))
    assert r.status_code == 200
    body = r.json()
    assert body["account"]["equity"] == 12_345.67
    assert body["positions"][0]["symbol"] == "AAPL"
    assert body["today"]["n_decisions"] >= 1


def test_halt_creates_kill_switch_and_resume_removes_it(
    client, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tc, _, _ = client
    monkeypatch.chdir(tmp_path)  # KILL_SWITCH lives in cwd
    assert not (tmp_path / "KILL_SWITCH").exists()

    r = tc.post("/control/halt", auth=(_USER, _PASS), follow_redirects=False)
    assert r.status_code == 303
    assert (tmp_path / "KILL_SWITCH").exists()

    r = tc.post("/control/resume", auth=(_USER, _PASS), follow_redirects=False)
    assert r.status_code == 303
    assert not (tmp_path / "KILL_SWITCH").exists()


def test_flatten_cancels_and_sells_then_halts(
    client, tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    tc, broker, _ = client
    monkeypatch.chdir(tmp_path)

    r = tc.post("/control/flatten", auth=(_USER, _PASS), follow_redirects=False)
    assert r.status_code == 303
    assert broker.cancelled_all == 1
    # Stub had one open AAPL long → one SELL submitted.
    assert len(broker.orders_submitted) == 1
    assert broker.orders_submitted[0].symbol == "AAPL"
    # Flatten halts new orders too.
    assert (tmp_path / "KILL_SWITCH").exists()


def test_dashboard_503s_when_no_password_configured(tmp_path: Path) -> None:
    settings = Settings(
        alpaca_api_key=SecretStr("k"),
        alpaca_secret_key=SecretStr("s"),
        anthropic_api_key=SecretStr("ak"),
        aitrade_dashboard_password=SecretStr(""),
        aitrade_data_dir=tmp_path / "data",
        aitrade_log_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    app = build_app(settings=settings, broker_factory=lambda: None)
    from aitrade.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    tc = TestClient(app)
    r = tc.get("/", auth=("u", "p"))
    assert r.status_code == 503
    assert "not configured" in r.text.lower()
