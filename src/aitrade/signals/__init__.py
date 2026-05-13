"""Conviction scoring: multi-component signal aggregation with calibrated weights.

Each component implements a uniform contract — see
``aitrade.signals.components.base.Scorer`` — and returns
``{score, weight, evidence, features}``. The composite is a weighted sum
fitted on historical out-of-sample data by ``aitrade.signals.calibration``.

The same JSON serves UI rendering, LLM prompts (via ``evidence``), and
backtest training (via ``features``).
"""

from aitrade.signals.conviction import (
    ConvictionComponent,
    ConvictionSnapshot,
    composite_from_components,
)

__all__ = [
    "ConvictionComponent",
    "ConvictionSnapshot",
    "composite_from_components",
]
