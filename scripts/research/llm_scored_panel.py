"""Compare LlmScoredBollinger to base BollingerReversion on the 5-symbol panel.

Single-pass (not walk-forward) on 2024-2026 daily bars. Same series for
both strategies — so any difference comes purely from the LLM gate.

For each (strategy, symbol):
  - run sequentially through all bars
  - feed signals to a tiny ledger that tracks entry/exit and P&L
  - report: trades taken, win rate, mean P&L per trade, total P&L,
    LLM-rejection rate (scored variant only)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import fmean

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Bar, Timeframe
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.examples.llm_scored_bollinger import LlmScoredBollinger
from aitrade.strategy.signal import Direction

SYMBOLS = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "PLTR", "MSFT", "BA", "MU"]
NOTIONAL = 10_000.0


@dataclass
class Trade:
    entry_ts: str
    entry_price: float
    exit_ts: str
    exit_price: float

    @property
    def ret_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0

    @property
    def pnl_usd(self) -> float:
        qty = NOTIONAL // self.entry_price
        return qty * (self.exit_price - self.entry_price)


@dataclass
class Ledger:
    trades: list[Trade] = field(default_factory=list)
    _entry_bar: Bar | None = None

    def step(self, bar: Bar, direction: Direction) -> None:
        if direction is Direction.LONG and self._entry_bar is None:
            self._entry_bar = bar
        elif direction is Direction.FLAT and self._entry_bar is not None:
            self.trades.append(
                Trade(
                    entry_ts=str(self._entry_bar.timestamp.date()),
                    entry_price=self._entry_bar.close,
                    exit_ts=str(bar.timestamp.date()),
                    exit_price=bar.close,
                )
            )
            self._entry_bar = None


def run(strategy_factory, bars: list[Bar]) -> Ledger:
    strat = strategy_factory()
    ledger = Ledger()
    for bar in bars:
        sig = strat.on_bar(bar)
        if sig is not None:
            ledger.step(bar, sig.direction)
    return ledger, strat


def summarize(ledger: Ledger) -> dict[str, float]:
    if not ledger.trades:
        return {"n": 0, "win_pct": 0, "mean_pnl": 0, "total_pnl": 0,
                "mean_ret_pct": 0}
    rets = [t.ret_pct for t in ledger.trades]
    pnls = [t.pnl_usd for t in ledger.trades]
    return {
        "n": len(ledger.trades),
        "win_pct": (sum(1 for r in rets if r > 0) / len(rets)) * 100.0,
        "mean_pnl": fmean(pnls),
        "total_pnl": sum(pnls),
        "mean_ret_pct": fmean(rets),
    }


client = AlpacaDataClient(get_settings())
results: list[tuple[str, dict[str, float], dict[str, float], int, int]] = []

for sym in SYMBOLS:
    df = client.fetch_stock_bars(
        sym, Timeframe.DAY_1,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    if df.empty:
        continue
    bars: list[Bar] = list(df_to_bars(df, sym))

    print(f"\n--- {sym} ---")
    base_ledger, _ = run(
        lambda sym=sym: BollingerReversion(symbol=sym, period=20, num_std=1.5),
        bars,
    )
    base = summarize(base_ledger)
    print(f"  BASE     trades={base['n']:>3} win={base['win_pct']:>5.1f}%  "
          f"mean_pnl=${base['mean_pnl']:>+7.2f}  total=${base['total_pnl']:>+8.2f}")

    scored_ledger, scored_strat = run(
        lambda sym=sym: LlmScoredBollinger(
            symbol=sym, period=20, num_std=1.5, min_score=7
        ),
        bars,
    )
    scored = summarize(scored_ledger)
    print(f"  SCORED≥7 trades={scored['n']:>3} win={scored['win_pct']:>5.1f}%  "
          f"mean_pnl=${scored['mean_pnl']:>+7.2f}  total=${scored['total_pnl']:>+8.2f}  "
          f"(approve={scored_strat.n_approve} reject={scored_strat.n_reject})")

    results.append((sym, base, scored, scored_strat.n_approve, scored_strat.n_reject))

print("\n=== PANEL SUMMARY ===")
print(f"{'sym':<6} {'base_n':>6} {'base_win%':>10} {'base_pnl':>10}  "
      f"{'sc_n':>5} {'sc_win%':>8} {'sc_pnl':>10}  {'delta_pnl':>10}")
total_base = 0.0
total_scored = 0.0
for sym, base, scored, _, _ in results:
    delta = scored["total_pnl"] - base["total_pnl"]
    total_base += base["total_pnl"]
    total_scored += scored["total_pnl"]
    print(f"{sym:<6} {base['n']:>6d} {base['win_pct']:>9.1f}% "
          f"${base['total_pnl']:>+9.2f}  "
          f"{scored['n']:>5d} {scored['win_pct']:>7.1f}% "
          f"${scored['total_pnl']:>+9.2f}  ${delta:>+9.2f}")
print(f"{'TOT':<6} {'':>6} {'':>10} ${total_base:>+9.2f}  "
      f"{'':>5} {'':>8} ${total_scored:>+9.2f}  ${total_scored - total_base:>+9.2f}")
