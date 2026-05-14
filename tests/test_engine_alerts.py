"""Phase 6 engine wiring — daily-loss alerts, time-gated skip behavior."""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.alerts import AlertLevel, Notifier
from aitrade.bots.engine_runner import (
    _DAILY_LOSS_ALERTED,
    EngineConfig,
    _maybe_alert_daily_loss,
)


@dataclass
class _FakeRisk:
    """Duck-typed RiskGate stand-in — exposes only the fields the alerter reads."""

    _daily_realized_pnl: float
    max_daily_loss_usd: float


class _CapturingHTTP:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object] | None = None) -> object:
        self.calls.append(dict(json or {}))

        class R:
            status_code = 200

        return R()


def _notifier() -> tuple[Notifier, _CapturingHTTP]:
    stub = _CapturingHTTP()
    n = Notifier(
        webhook_url="https://example.test/hook",
        min_level=AlertLevel.INFO,
        http=stub,
    )
    return n, stub


def _cfg() -> EngineConfig:
    return EngineConfig(
        daily_loss_warn_pct=0.5, daily_loss_error_pct=0.9
    )


def setup_function() -> None:
    """Reset the module-level alert dedup set between tests."""
    _DAILY_LOSS_ALERTED.clear()


def test_no_alert_when_pnl_positive() -> None:
    n, stub = _notifier()
    risk = _FakeRisk(_daily_realized_pnl=50.0, max_daily_loss_usd=500.0)
    _maybe_alert_daily_loss(risk, n, _cfg())
    assert stub.calls == []


def test_warn_fires_at_50pct_threshold() -> None:
    n, stub = _notifier()
    risk = _FakeRisk(_daily_realized_pnl=-260.0, max_daily_loss_usd=500.0)  # 52%
    _maybe_alert_daily_loss(risk, n, _cfg())
    assert len(stub.calls) == 1
    assert stub.calls[0]["level"] == "warn"


def test_warn_fires_only_once_per_session() -> None:
    n, stub = _notifier()
    risk = _FakeRisk(_daily_realized_pnl=-260.0, max_daily_loss_usd=500.0)
    _maybe_alert_daily_loss(risk, n, _cfg())
    _maybe_alert_daily_loss(risk, n, _cfg())  # threshold still crossed
    _maybe_alert_daily_loss(risk, n, _cfg())
    assert len(stub.calls) == 1


def test_error_fires_at_90pct_after_warn() -> None:
    n, stub = _notifier()
    risk = _FakeRisk(_daily_realized_pnl=-260.0, max_daily_loss_usd=500.0)  # 52%
    _maybe_alert_daily_loss(risk, n, _cfg())
    risk._daily_realized_pnl = -460.0  # 92%
    _maybe_alert_daily_loss(risk, n, _cfg())
    levels = [c["level"] for c in stub.calls]
    assert levels == ["warn", "error"]


def test_no_alert_with_zero_cap() -> None:
    """Cap of 0 disables daily-loss alerting entirely."""
    n, stub = _notifier()
    risk = _FakeRisk(_daily_realized_pnl=-9999.0, max_daily_loss_usd=0.0)
    _maybe_alert_daily_loss(risk, n, _cfg())
    assert stub.calls == []
