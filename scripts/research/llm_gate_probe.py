"""Probe: what is the LLM actually saying at each Bollinger entry?

Runs LlmGatedBollinger sequentially through the full 2024-2026 daily
series on each symbol (no walkforward, no fresh-instance per fold —
ONE strategy across the whole range). Prints every LLM decision: the
bar timestamp, the close, the inner Bollinger reason, and the verbatim
LLM response text.

Used to diagnose whether the gate is rejecting too much, approving too
much, or hitting silent errors. Run on a tight panel first to bound
LLM cost (~$0.05).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

from loguru import logger

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe
from aitrade.strategy.examples.llm_gated_bollinger import LlmGatedBollinger
from aitrade.strategy.signal import Direction

SYMBOLS = sys.argv[1:] if len(sys.argv) > 1 else ["AAPL", "PLTR", "MSFT", "BA", "MU"]

# Drop loguru below INFO so the LLM debug lines surface cleanly.
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="{message}")

client = AlpacaDataClient(get_settings())


def probe(sym: str) -> None:
    df = client.fetch_stock_bars(
        sym,
        Timeframe.DAY_1,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    if df.empty:
        print(f"[{sym}] no bars")
        return
    bars = list(df_to_bars(df, sym))
    strat = LlmGatedBollinger(symbol=sym, period=20, num_std=1.5)

    decisions: list[tuple[str, float, str, str]] = []
    for bar in bars:
        prev_approve = strat.n_approve
        prev_reject = strat.n_reject
        prev_error = strat.n_error
        sig = strat.on_bar(bar)
        if (
            strat.n_approve != prev_approve
            or strat.n_reject != prev_reject
            or strat.n_error != prev_error
        ):
            if strat.n_approve != prev_approve:
                outcome = "APPROVE"
            elif strat.n_reject != prev_reject:
                outcome = "REJECT "
            else:
                outcome = "ERROR  "
            ts = str(bar.timestamp.date())
            decisions.append((ts, bar.close, outcome, strat._last_decision or ""))
        # Also catch signals that go through default-approve (insufficient history).
        if sig is not None and sig.direction is Direction.LONG and not decisions:
            decisions.append((str(bar.timestamp.date()), bar.close, "DEFAULT", "(<21 bars)"))

    print(f"\n=== {sym} ===")
    print(
        f"  total LLM-evaluated entries: "
        f"approve={strat.n_approve} reject={strat.n_reject} error={strat.n_error}"
    )
    if not decisions:
        print("  no entries triggered at all (inner bollinger never fired)")
        return
    for ts, close, outcome, text in decisions:
        first_line = text.split("\n")[0][:120] if text else ""
        print(f"  {ts}  ${close:>7.2f}  {outcome}  {first_line}")


for sym in SYMBOLS:
    probe(sym)
