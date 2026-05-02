"""Economic calendar — Financial Modeling Prep client with TTL cache.

Exposes a compact, prompt-ready projection of upcoming US macro events
(Fed/FOMC, CPI, NFP, GDP, retail sales, ...) and earnings announcements.
Failures are non-load-bearing: every public method returns ``[]`` on error.
"""

from __future__ import annotations

from aitrade.calendar.client import EarningsEvent, EconomicCalendarClient, EconomicEvent

__all__ = ["EarningsEvent", "EconomicCalendarClient", "EconomicEvent"]
