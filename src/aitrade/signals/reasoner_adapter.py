"""Adapter: real Reasoner → AcceptFn for the value-test harness.

Builds a ReasonerInput from a setup row + the bars window that produced
it, calls the Reasoner, and reports back accept/reject based on
``should_trade`` AND a configurable confidence threshold.

Kept in its own module so ``signals.value_test`` stays free of the
heavy reasoning import surface for unit tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from aitrade.data.models import Bar
from aitrade.reasoning.claude_reasoner import Reasoner
from aitrade.reasoning.decision import ReasonerInput
from aitrade.strategy.indicators import compute_snapshot


@dataclass(frozen=True, slots=True)
class ReasonerAcceptConfig:
    cash_available: float = 100_000.0
    buying_power: float = 100_000.0
    max_position_usd: float = 10_000.0
    confidence_floor: float = 0.6


def make_accept_fn(
    reasoner: Reasoner,
    bars_by_ts: dict[pd.Timestamp, Sequence[Bar]],
    *,
    config: ReasonerAcceptConfig | None = None,
) -> Callable[[pd.Series], bool]:
    """Build an AcceptFn compatible with ``value_test.value_test``.

    Args:
        reasoner: any Reasoner-Protocol instance (real Claude, mock, etc.).
        bars_by_ts: setup timestamp → bars window at that time. Populated by
            the caller during ``collect_setups`` if it stashes windows alongside.
        config: per-call hypothetical account params + confidence floor.

    Returns a callable matching ``signals.value_test.AcceptFn``.
    """
    cfg = config or ReasonerAcceptConfig()

    def accept(row: pd.Series) -> bool:
        ts = pd.Timestamp(row["ts"])
        bars = bars_by_ts.get(ts)
        if not bars:
            return False
        snap = compute_snapshot(list(bars))
        decision = reasoner.decide(
            ReasonerInput(
                symbol=bars[-1].symbol,
                indicators=snap,
                cash_available=cfg.cash_available,
                buying_power=cfg.buying_power,
                max_position_usd=cfg.max_position_usd,
            )
        )
        if decision is None:
            return False
        return bool(decision.should_trade and decision.confidence >= cfg.confidence_floor)

    return accept
