"""Conviction calibration: information coefficient + weight fitting.

Workflow:

1. Replay bars chronologically; at each "setup" (e.g. when a strategy
   emits a non-flat signal), record each component's score + the
   N-bar-forward realized return.
2. Compute Spearman IC per component on out-of-sample fold.
3. Fit composite weights to maximize correlation with forward returns
   subject to non-negativity + sum-to-one, via grid search.

No new dependencies; weight fitting uses a coarse grid that's adequate
for 4-component vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import pandas as pd

from aitrade.data.models import Bar
from aitrade.signals.components.base import Scorer, ScoringContext
from aitrade.signals.conviction import ConvictionComponent
from aitrade.strategy.base import Strategy

StrategyBuilder = Callable[[str], Strategy]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    ic_per_component: dict[str, float]  # Spearman IC vs forward return
    fitted_weights: dict[str, float]  # non-negative, sum to 1
    composite_ic: float  # Spearman IC of weighted composite
    n_setups: int


def _forward_returns(closes: pd.Series, horizon: int) -> pd.Series:
    return closes.shift(-horizon) / closes - 1.0


def _grid_weights(n: int, step: float = 0.1) -> list[list[float]]:
    """Enumerate weight vectors of length n with values on a step-grid summing to 1.0."""
    levels = [round(i * step, 6) for i in range(int(1.0 / step) + 1)]
    out: list[list[float]] = []
    for combo in product(levels, repeat=n):
        if abs(sum(combo) - 1.0) < 1e-9:
            out.append(list(combo))
    return out


def collect_setups(
    strategy_builder: StrategyBuilder | None,
    bars: Sequence[Bar],
    scorers: Sequence[Scorer],
    *,
    horizon: int = 5,
    warmup: int = 60,
) -> pd.DataFrame:
    """Walk bars chronologically; at each accepted setup compute components + forward return.

    Args:
        strategy_builder: optional callable ``symbol -> Strategy``. If given,
            only bars where the strategy emits a non-flat signal become setups.
            If None, every post-warmup bar is a setup (useful for measuring
            scorer IC independent of a strategy's gating).
    """
    if len(bars) <= warmup + horizon:
        return pd.DataFrame()

    strategy: Strategy | None = (
        strategy_builder(bars[0].symbol) if strategy_builder is not None else None
    )

    closes = pd.Series(
        [b.close for b in bars],
        index=[b.timestamp for b in bars],
        dtype=float,
    )
    fwd = _forward_returns(closes, horizon)

    rows: list[dict[str, float | pd.Timestamp]] = []
    window: list[Bar] = []
    for i, bar in enumerate(bars):
        window.append(bar)
        if i < warmup:
            if strategy is not None:
                strategy.on_bar(bar)
            continue

        signal = strategy.on_bar(bar) if strategy is not None else None
        # If a strategy is supplied, only accept setups when it emits a non-flat
        # signal. Otherwise accept every post-warmup bar.
        from aitrade.strategy.signal import Direction

        if strategy is not None and (signal is None or signal.direction is Direction.FLAT):
            continue
        if i + horizon >= len(bars):
            break
        if pd.isna(fwd.iloc[i]):
            continue

        ctx = ScoringContext(bars=window[-warmup:], signal=signal)
        row: dict[str, float | pd.Timestamp] = {
            "ts": bar.timestamp,
            "fwd_return": float(fwd.iloc[i]),
        }
        for s in scorers:
            row[s.name] = s.score(ctx).score
        rows.append(row)

    return pd.DataFrame(rows)


def fit_weights(
    setups: pd.DataFrame,
    scorer_names: Sequence[str],
    *,
    grid_step: float = 0.1,
) -> CalibrationResult:
    """Fit non-negative, sum-to-one weights maximizing Spearman IC."""
    if setups.empty:
        return CalibrationResult(
            ic_per_component={n: 0.0 for n in scorer_names},
            fitted_weights={n: 1.0 / len(scorer_names) for n in scorer_names},
            composite_ic=0.0,
            n_setups=0,
        )

    fwd = setups["fwd_return"]
    ic_per_component: dict[str, float] = {}
    for n in scorer_names:
        col = setups[n]
        if col.std() == 0:
            ic_per_component[n] = 0.0
            continue
        ic_per_component[n] = float(col.rank().corr(fwd.rank()))

    best_ic = -2.0
    best_w: list[float] = [1.0 / len(scorer_names)] * len(scorer_names)
    for w in _grid_weights(len(scorer_names), step=grid_step):
        composite = sum(w[i] * setups[scorer_names[i]] for i in range(len(scorer_names)))
        if composite.std() == 0:
            continue
        ic = float(composite.rank().corr(fwd.rank()))
        if ic > best_ic:
            best_ic = ic
            best_w = w

    return CalibrationResult(
        ic_per_component=ic_per_component,
        fitted_weights={scorer_names[i]: best_w[i] for i in range(len(scorer_names))},
        composite_ic=best_ic if best_ic > -2.0 else 0.0,
        n_setups=len(setups),
    )


def build_components(
    scorers: Sequence[Scorer],
    ctx: ScoringContext,
    weight_overrides: dict[str, float] | None = None,
) -> dict[str, ConvictionComponent]:
    """Run every scorer; apply weight overrides if given."""
    out: dict[str, ConvictionComponent] = {}
    for s in scorers:
        c = s.score(ctx)
        if weight_overrides and s.name in weight_overrides:
            c = ConvictionComponent(
                score=c.score,
                weight=weight_overrides[s.name],
                evidence=c.evidence,
                features=c.features,
            )
        out[s.name] = c
    return out
