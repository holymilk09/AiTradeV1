"""Local out-of-band notifications. macOS `osascript` only; no external services."""

from aitrade.notify.macos import Urgency, send

__all__ = ["Urgency", "send"]
