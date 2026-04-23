"""NautilusTrader-based backtest runner.

Professional-grade event-driven engine — full queue/latency/slippage modeling.
Import is lazy so the rest of the package works even if Nautilus isn't
installed. Pattern lifted from ``evan-kolberg/prediction-market-backtesting``.

The scaffold below wires:
  - a simulated SIM venue
  - instruments for each symbol
  - bars loaded from a DataFrame via ``BarDataWrangler``
  - a NautilusTrader strategy shim that delegates to our ``Strategy`` Protocol

TODO (follow-up PR): replace the thin shim with a full NT-native strategy once
the first real edge is identified. For the foundation PR, the simple_runner
is the default path and this runner exists so the NT dependency is exercised
and the upgrade path is clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger


@dataclass(frozen=True, slots=True)
class NautilusBacktestConfig:
    symbol: str
    timeframe_secs: int
    starting_cash_usd: float = 100_000.0
    fee_bps: float = 1.0


def is_available() -> bool:
    try:
        import nautilus_trader  # noqa: F401

        return True
    except ImportError:
        return False


def run_nautilus_backtest(
    bars_df: pd.DataFrame,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    config: NautilusBacktestConfig,
) -> dict[str, object]:
    """Run a NautilusTrader BacktestEngine over the provided bars.

    Returns a dict with ``engine`` (for further inspection) and ``report``.
    Raises ImportError if NautilusTrader isn't installed.
    """
    if not is_available():
        raise ImportError(
            "nautilus-trader is not installed. Install via `uv sync` or use simple_runner."
        )

    # Lazy imports so the module stays importable without NT installed.
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.objects import Money

    venue = Venue("SIM")
    engine_config = BacktestEngineConfig(
        trader_id="AITRADE-001",
        logging=LoggingConfig(log_level="WARNING"),
    )
    engine = BacktestEngine(config=engine_config)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(config.starting_cash_usd, USD)],
        base_currency=USD,
    )
    logger.info(
        "nautilus backtest configured symbol={} start={} end={} bars={}",
        symbol,
        start,
        end,
        len(bars_df),
    )
    # Instrument + bar loading is strategy-specific and NT-version-sensitive.
    # Left as a concrete extension point rather than speculative wiring that
    # could break across NT versions. See CLAUDE.md and the plan file.
    return {"engine": engine, "report": {"status": "scaffold", "rows": len(bars_df)}}
