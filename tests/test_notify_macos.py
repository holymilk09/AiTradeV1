"""macOS notify helper: portable no-op on non-darwin, escape-correct."""

from __future__ import annotations

from unittest.mock import patch

from aitrade.notify.macos import Urgency, _escape, send


def test_noop_on_non_darwin() -> None:
    with patch("aitrade.notify.macos._is_darwin", return_value=False):
        assert send("t", "b") is False


def test_escapes_quotes_and_backslashes() -> None:
    assert _escape('he said "hi"') == 'he said \\"hi\\"'
    assert _escape("path\\to") == "path\\\\to"


def test_send_invokes_osascript_on_darwin() -> None:
    with (
        patch("aitrade.notify.macos._is_darwin", return_value=True),
        patch("aitrade.notify.macos.shutil.which", return_value="/usr/bin/osascript"),
        patch("aitrade.notify.macos.subprocess.run") as mock_run,
    ):
        assert send("title", "body", urgency=Urgency.CRITICAL) is True
        assert mock_run.called
        args = mock_run.call_args.args[0]
        assert args[0] == "/usr/bin/osascript"
        assert "display notification" in args[2]
        assert "Glass" in args[2]  # critical sound


def test_send_handles_missing_osascript() -> None:
    with (
        patch("aitrade.notify.macos._is_darwin", return_value=True),
        patch("aitrade.notify.macos.shutil.which", return_value=None),
    ):
        assert send("t", "b") is False
