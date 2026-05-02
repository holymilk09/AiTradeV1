"""Mobile-friendly FastAPI dashboard for the bot.

Exposes a small password-protected web UI that you can open on your phone
to check the bot's status, see today's fills, view the round-trip ledger,
and hit a kill switch — all without SSH'ing into the box.

Designed for self-hosting on fly.io / DigitalOcean / Railway alongside
the engine, with a single-process ``aitrade serve`` that runs both.
"""

from aitrade.dashboard.app import build_app

__all__ = ["build_app"]
