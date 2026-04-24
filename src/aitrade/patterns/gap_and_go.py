"""Intraday gap-up + hold detector for 5-minute bars."""

from __future__ import annotations

from dataclasses import dataclass

from aitrade.data.models import Bar
from aitrade.patterns.base import PatternSignal, clamp01
from aitrade.strategy.signal import Direction


def _session_open_index(bars: list[Bar]) -> int | None:
    """Return the index of the first bar of the most recent session.

    The first bar of "today" is the latest bar whose calendar date differs
    from the bar before it. Returns ``None`` if no such boundary exists in
    the supplied window.
    """
    for i in range(len(bars) - 1, 0, -1):
        if bars[i].timestamp.date() != bars[i - 1].timestamp.date():
            return i
    return None


@dataclass(frozen=True, slots=True)
class GapAndGoDetector:
    """Fires when today's open gaps up and holds for ``hold_mins`` minutes.

    Designed for 5-minute intraday bars. The detector finds today's first
    bar, checks that its open exceeds the previous day's last close by
    ``gap_pct``, and then verifies that every subsequent bar within
    ``hold_mins`` minutes closed at-or-above that gap-bar open.
    """

    name: str = "gap_and_go"
    gap_pct: float = 1.02
    hold_mins: int = 15

    def detect(self, bars: list[Bar]) -> PatternSignal | None:
        """Return a LONG signal when the open gap is held for ``hold_mins`` minutes."""
        if self.gap_pct <= 1.0 or self.hold_mins <= 0:
            return None
        if len(bars) < 2:
            return None

        open_idx = _session_open_index(bars)
        if open_idx is None or open_idx == 0:
            return None

        gap_bar = bars[open_idx]
        prev_close = bars[open_idx - 1].close
        if prev_close <= 0:
            return None
        if gap_bar.open <= prev_close * self.gap_pct:
            return None

        # Inspect every bar strictly after the open bar that falls within the
        # ``hold_mins`` window. We need at least one confirming bar.
        held: list[Bar] = []
        for bar in bars[open_idx + 1 :]:
            elapsed = (bar.timestamp - gap_bar.timestamp).total_seconds() / 60.0
            if elapsed > self.hold_mins:
                break
            held.append(bar)

        if not held:
            return None
        if any(b.close < gap_bar.open for b in held):
            return None

        gap_size_pct = (gap_bar.open - prev_close) / prev_close
        held_mins = (held[-1].timestamp - gap_bar.timestamp).total_seconds() / 60.0
        score = clamp01(gap_size_pct * 30.0)

        return PatternSignal(
            name=self.name,
            symbol=gap_bar.symbol,
            score=score,
            direction=Direction.LONG,
            timestamp=held[-1].timestamp,
            evidence={
                "gap_pct": gap_size_pct,
                "held_mins": held_mins,
                "session_open": gap_bar.open,
            },
        )
