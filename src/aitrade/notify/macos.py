"""macOS local notification via osascript.

Fire-and-forget. Never raises — notifications are advisory. The source
of truth for any event is always the trade journal; this surface just
nudges the operator when they're away from the dashboard.

No-op on non-darwin platforms so tests + CI stay portable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from enum import Enum

from loguru import logger


class Urgency(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _escape(s: str) -> str:
    """Escape double quotes + backslashes for AppleScript string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def send(title: str, body: str, urgency: Urgency = Urgency.INFO) -> bool:
    """Display a macOS notification. Returns True on best-effort dispatch."""
    if not _is_darwin():
        logger.debug("notify skipped (non-darwin) title={!r}", title)
        return False
    osascript = shutil.which("osascript")
    if osascript is None:
        logger.warning("osascript not on PATH; notifications disabled")
        return False
    sound = "Glass" if urgency is Urgency.CRITICAL else "Pop"
    script = (
        f'display notification "{_escape(body)}" '
        f'with title "{_escape(title)}" sound name "{sound}"'
    )
    try:
        subprocess.run(
            [osascript, "-e", script],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("notify dispatch failed: {}", e)
        return False
    return True
