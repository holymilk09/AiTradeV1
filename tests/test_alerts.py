"""Generic webhook notifier — level gating + best-effort failure modes."""

from __future__ import annotations

from aitrade.alerts import AlertLevel, Notifier, build_notifier


class _StubResp:
    def __init__(self, status: int = 200) -> None:
        self.status_code = status


class _StubHTTP:
    """Captures every POST so tests can assert wire format + call count."""

    def __init__(self, status: int = 200, raise_on_post: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.status = status
        self.raise_on_post = raise_on_post

    def post(self, url: str, json: dict[str, object] | None = None) -> _StubResp:
        if self.raise_on_post:
            raise RuntimeError("boom")
        self.calls.append((url, dict(json or {})))
        return _StubResp(self.status)


def _notifier(
    *,
    url: str | None = "https://example.test/hook",
    min_level: AlertLevel = AlertLevel.INFO,
    http: _StubHTTP | None = None,
) -> tuple[Notifier, _StubHTTP]:
    stub = http or _StubHTTP()
    n = Notifier(webhook_url=url, min_level=min_level, http=stub)
    return n, stub


def test_disabled_when_no_url() -> None:
    n, stub = _notifier(url="")
    assert n.is_enabled is False
    assert n.info("hello") is False
    assert stub.calls == []  # never even tries to POST


def test_send_payload_shape_matches_slack_discord_ntfy() -> None:
    """Wire format: {"text": "...", "level": "info|warn|error"} works on
    Slack incoming webhooks, Discord webhooks, and ntfy."""
    n, stub = _notifier()
    assert n.warn("daily loss approaching") is True
    assert len(stub.calls) == 1
    url, payload = stub.calls[0]
    assert url == "https://example.test/hook"
    assert payload == {"text": "daily loss approaching", "level": "warn"}


def test_min_level_drops_lower_severity() -> None:
    n, stub = _notifier(min_level=AlertLevel.WARN)
    assert n.info("routine") is False  # below threshold → no POST
    assert n.warn("watch out") is True
    assert n.error("on fire") is True
    assert len(stub.calls) == 2  # info dropped


def test_post_failure_returns_false_does_not_raise() -> None:
    """Webhook outages must never break the engine."""
    n, _ = _notifier(http=_StubHTTP(raise_on_post=True))
    assert n.error("alpaca down") is False  # swallowed


def test_non_2xx_response_returns_false() -> None:
    n, _ = _notifier(http=_StubHTTP(status=503))
    assert n.info("hi") is False


def test_alert_level_parse_round_trips() -> None:
    assert AlertLevel.parse("info") is AlertLevel.INFO
    assert AlertLevel.parse("WARN") is AlertLevel.WARN
    assert AlertLevel.parse("Error") is AlertLevel.ERROR
    # Unknown / blank → INFO (the safe default).
    assert AlertLevel.parse("nonsense") is AlertLevel.INFO
    assert AlertLevel.parse("") is AlertLevel.INFO


def test_build_notifier_factory_is_noop_when_no_url() -> None:
    """Factory used by the CLI should never throw and should produce a
    noop when the operator hasn't opted in."""
    n = build_notifier(webhook_url="", min_level="warn")
    assert n.is_enabled is False
    assert n.error("anything") is False


def test_build_notifier_honors_min_level_string() -> None:
    n = build_notifier(webhook_url="https://x.test/y", min_level="warn")
    # We can't introspect _min_level directly without breaking encapsulation,
    # but we CAN observe behavior: an info call should drop, warn should fire.
    stub = _StubHTTP()
    n._http = stub  # type: ignore[attr-defined]
    assert n.info("routine") is False
    assert n.warn("watch") is True
    assert len(stub.calls) == 1
