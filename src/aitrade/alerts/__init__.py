"""Generic webhook notifier for push alerts.

Designed so the bot can run unattended overnight and still wake the
operator when something matters. The wire format is the lowest common
denominator across Slack, Discord, and ntfy:

    POST <webhook_url>
    Content-Type: application/json
    {"text": "...", "level": "info|warn|error"}

Slack and Discord both happily accept ``{"text": "..."}`` (Slack's
``incoming-webhook`` shape; Discord interprets ``text`` as plain content
when no embed is present). For ntfy, the payload becomes the message
body; the ``level`` field is ignored harmlessly. If you need richer
formatting per-vendor, swap ``Notifier`` for a vendor-specific subclass —
the engine reaches it through a thin ``notify()`` interface.

Levels and gating
-----------------
The notifier honors a *minimum level* (default ``info``). Calls below the
threshold are dropped silently — useful when running paper for days and
you only want push notifications on warns/errors. Levels are ordered
``info < warn < error``.

Failure mode
------------
Webhook posts are best-effort. Any HTTP error, timeout, or malformed URL
yields a logged warning and a ``False`` return — the engine never raises
into the trading loop because Slack happens to be down. When no URL is
configured at all, the notifier is a no-op.
"""

from __future__ import annotations

from enum import IntEnum

import httpx
from loguru import logger

_TIMEOUT_SECS = 5.0


class AlertLevel(IntEnum):
    """Severity ordering for notifier gating."""

    INFO = 10
    WARN = 20
    ERROR = 30

    @classmethod
    def parse(cls, raw: str) -> AlertLevel:
        """Parse a case-insensitive level name; fall back to INFO."""
        normalized = (raw or "").strip().lower()
        for level in cls:
            if level.name.lower() == normalized:
                return level
        return cls.INFO


class Notifier:
    """Generic webhook notifier.

    Parameters
    ----------
    webhook_url:
        The Slack/Discord/ntfy/custom incoming-webhook URL. Empty string
        or None disables all sends — every ``notify()`` call is a no-op
        (still returns ``False``). This is the default in tests + when
        the operator hasn't opted in.
    min_level:
        Drop calls below this severity. Default is ``INFO`` so callers see
        every event; bump to ``WARN`` to silence routine pings.
    http:
        Inject a stub for tests. Real client built lazily so importing the
        module is cheap.
    """

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        min_level: AlertLevel = AlertLevel.INFO,
        http: object | None = None,
    ) -> None:
        self._url = (webhook_url or "").strip()
        self._min_level = min_level
        self._http: object | None = http

    @property
    def is_enabled(self) -> bool:
        """True iff a webhook URL was configured."""
        return bool(self._url)

    def _get_http(self) -> object:
        if self._http is None:
            self._http = httpx.Client(timeout=_TIMEOUT_SECS)
        return self._http

    def notify(self, text: str, *, level: AlertLevel = AlertLevel.INFO) -> bool:
        """POST a single notification. Returns True on a 2xx response.

        Best-effort: any error path returns False without raising. When
        the notifier is disabled (no URL) or the level is below the
        configured threshold, returns False without contacting the
        network.
        """
        if not self.is_enabled:
            return False
        if int(level) < int(self._min_level):
            return False

        payload = {"text": text, "level": level.name.lower()}
        http = self._get_http()
        try:
            resp = http.post(self._url, json=payload)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("notifier post failed level={} err={}", level.name, exc)
            return False
        status = getattr(resp, "status_code", 0)
        ok = 200 <= status < 300
        if not ok:
            logger.warning(
                "notifier non-2xx response level={} status={}", level.name, status
            )
        return ok

    # Convenience helpers — semantically equivalent to notify() with a
    # specific level, just shorter at the call site.
    def info(self, text: str) -> bool:
        return self.notify(text, level=AlertLevel.INFO)

    def warn(self, text: str) -> bool:
        return self.notify(text, level=AlertLevel.WARN)

    def error(self, text: str) -> bool:
        return self.notify(text, level=AlertLevel.ERROR)


def build_notifier(webhook_url: str | None, min_level: str = "info") -> Notifier:
    """Convenience factory used by the CLI / engine wiring.

    Returns a no-op notifier when the URL is empty so callers never need
    to None-guard at every send site.
    """
    return Notifier(
        webhook_url=webhook_url,
        min_level=AlertLevel.parse(min_level),
    )


__all__ = [
    "AlertLevel",
    "Notifier",
    "build_notifier",
]
