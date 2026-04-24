"""Pattern detector registry — single place to enumerate all detectors."""

from __future__ import annotations

from aitrade.patterns.base import PatternDetector
from aitrade.patterns.breakout import BreakoutDetector
from aitrade.patterns.gap_and_go import GapAndGoDetector
from aitrade.patterns.pullback import PullbackDetector
from aitrade.patterns.unusual_volume import UnusualVolumeDetector
from aitrade.patterns.volume_trend import VolumePriceTrendDetector


def get_all_detectors() -> list[PatternDetector]:
    """Return one fresh instance of every shipped detector, default-configured."""
    return [
        VolumePriceTrendDetector(),
        BreakoutDetector(),
        UnusualVolumeDetector(),
        PullbackDetector(),
        GapAndGoDetector(),
    ]
