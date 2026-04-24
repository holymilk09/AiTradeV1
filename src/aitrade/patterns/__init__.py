"""Pattern detectors — pure-function setups over recent bars.

Each detector implements :class:`PatternDetector` and emits at most one
:class:`PatternSignal` per call. The registry's :func:`get_all_detectors`
returns a default-configured fan-out.
"""

from __future__ import annotations

from aitrade.patterns.base import PatternDetector, PatternSignal
from aitrade.patterns.breakout import BreakoutDetector
from aitrade.patterns.gap_and_go import GapAndGoDetector
from aitrade.patterns.pullback import PullbackDetector
from aitrade.patterns.registry import get_all_detectors
from aitrade.patterns.unusual_volume import UnusualVolumeDetector
from aitrade.patterns.volume_trend import VolumePriceTrendDetector

__all__ = [
    "BreakoutDetector",
    "GapAndGoDetector",
    "PatternDetector",
    "PatternSignal",
    "PullbackDetector",
    "UnusualVolumeDetector",
    "VolumePriceTrendDetector",
    "get_all_detectors",
]
