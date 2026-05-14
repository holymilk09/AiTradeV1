"""VIX-gated Bollinger reversion.

Wraps :class:`BollingerReversion` and gates *entries* on the
:class:`VixRegimeScorer` — only enters when the symbol's 20-day realized
vol is below ``rvol_pct_max`` (default 50, i.e. below-median vol regime).

Exits are NEVER gated. If the inner strategy says flat, we flatten —
volatility regime doesn't matter for getting out of a position.

Why this exists. The base bollinger_reversion strategy fires regardless
of regime, but EXP-004 showed the overnight-bias return profile flips
catastrophically in high-vol regimes. By gating entries the same way,
we expect:

  - higher per-symbol Sharpe (skip the bad-regime trades)
  - tighter Sharpe variance across symbols (consistency)
  - fewer trades — acceptable iff expectancy improves

The thesis is testable: if `consistency` improves and net expectancy
holds, the gate is a real upgrade. If it cuts too many good trades
(higher false-rejection rate than bad-regime save rate), the unfiltered
strategy is better. EXP-005 in docs/RESEARCH.md does that comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aitrade.data.models import Bar, Quote
from aitrade.execution.orders import Fill
from aitrade.signals.components import ScoringContext, VixRegimeScorer
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.signal import Direction, Signal

_BAR_HISTORY_CAP = 300  # bound memory; ~6 trading months on daily bars


@dataclass
class VixGatedBollinger:
    symbol: str
    period: int = 20
    num_std: float = 1.5
    rvol_pct_max: float = 50.0
    strategy_id: str = "vix_gated_bollinger"
    _bars: list[Bar] = field(init=False, default_factory=list)
    _inner: BollingerReversion = field(init=False)
    _scorer: VixRegimeScorer = field(init=False, default_factory=VixRegimeScorer)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rvol_pct_max <= 100.0:
            raise ValueError("rvol_pct_max must be in [0, 100]")
        self._inner = BollingerReversion(
            symbol=self.symbol, period=self.period, num_std=self.num_std
        )

    def on_bar(self, bar: Bar) -> Signal | None:
        if bar.symbol != self.symbol:
            return None

        self._bars.append(bar)
        if len(self._bars) > _BAR_HISTORY_CAP:
            self._bars = self._bars[-_BAR_HISTORY_CAP:]

        signal = self._inner.on_bar(bar)
        if signal is None:
            return None

        # Only gate entries (LONG). Exits (FLAT) always pass — never trap
        # a position because the regime turned ugly.
        if signal.direction is Direction.LONG:
            ctx = ScoringContext(bars=self._bars, signal=signal)
            component = self._scorer.score(ctx)
            rvol_pct = component.features.get("rvol_pct", 50.0)
            if rvol_pct > self.rvol_pct_max:
                return None

        return signal

    def on_quote(self, quote: Quote) -> Signal | None:
        return None

    def on_fill(self, fill: Fill) -> None:
        return None
