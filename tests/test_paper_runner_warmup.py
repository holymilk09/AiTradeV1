"""Paper runner warm-up: historical bars must NOT fire live orders.

Regression for the 9-orders-in-6-seconds bug. The runner refetches the
last 5 days of bars on every poll for state-warming; without a warm-up
guard, every historical SMA crossover in that window fires a real
order on first poll.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd

from aitrade.bots.runner import PaperRunConfig, run_paper_session
from aitrade.brokers.base import AccountSummary, BrokerClient, Position
from aitrade.data.models import Timeframe
from aitrade.execution.executor import Executor
from aitrade.execution.orders import OrderAck, OrderRequest, OrderStatus
from aitrade.execution.risk import RiskGate


@dataclass
class FakeBroker(BrokerClient):
    submitted: list[OrderRequest] = field(default_factory=list)

    @property
    def is_paper(self) -> bool:
        return True

    def get_account(self) -> AccountSummary:
        return AccountSummary(
            account_id="x", cash=100_000, equity=100_000, buying_power=100_000, is_paper=True
        )

    def get_positions(self) -> list[Position]:
        return []

    def submit_order(self, order: OrderRequest) -> OrderAck:
        self.submitted.append(order)
        return OrderAck(
            client_order_id=order.client_order_id,
            broker_order_id=f"br-{len(self.submitted)}",
            status=OrderStatus.ACCEPTED,
        )

    def cancel_order(self, broker_order_id: str) -> None:
        pass

    def cancel_all(self) -> int:
        return 0

    def is_tradable(self, symbol: str) -> bool:
        return True


def _bars_df(start_ts: datetime, n: int, slope: float = 0.5) -> pd.DataFrame:
    """Return n daily bars with a steady uptrend so SMA fast crosses slow."""
    rows = []
    for i in range(n):
        price = 100.0 + i * slope
        rows.append(
            {
                "timestamp": start_ts + timedelta(minutes=i),
                "open": price,
                "high": price + 0.1,
                "low": price - 0.1,
                "close": price,
                "volume": 1_000_000,
                "trade_count": 1,
                "vwap": price,
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")


def test_warmup_blocks_historical_signals_from_submitting() -> None:
    """All bars predate session_start → strategy warms on them but zero
    orders should reach the broker."""
    # 60 bars of clear uptrend would normally cross SMA(10) above SMA(30)
    # multiple times and fire LONG. They're all dated in the past.
    historical_start = datetime.now(UTC) - timedelta(days=5)
    df = _bars_df(historical_start, n=60)

    broker = FakeBroker()
    data = MagicMock()
    data.fetch_stock_bars.return_value = df

    risk = RiskGate(
        max_position_usd=1_000_000,
        max_daily_loss_usd=1_000_000,
        max_orders_per_min=999,
    )
    exe = Executor(broker, risk)

    journal = MagicMock()
    journal.run_id = "test"

    from aitrade.strategy.examples.sma_crossover import SmaCrossover

    strat = SmaCrossover(symbol="AAPL", fast=5, slow=20)

    # Run for a very short duration; we just need one poll.
    cfg = PaperRunConfig(
        symbol="AAPL",
        timeframe=Timeframe.MIN_1,
        duration=timedelta(seconds=0),
        poll_interval_secs=1,
        target_position_notional=2_000.0,
    )
    run_paper_session(strat, broker, data, exe, journal, cfg)

    # All 60 bars predate session_start → no orders should have been submitted.
    assert broker.submitted == []


def test_fetch_failure_does_not_crash_session() -> None:
    """A RemoteDisconnected (or any exception) from fetch_stock_bars must
    NOT propagate out of run_paper_session. The loop should log and continue."""
    broker = FakeBroker()
    data = MagicMock()
    data.fetch_stock_bars.side_effect = ConnectionError(
        "Connection aborted., RemoteDisconnected"
    )

    risk = RiskGate(
        max_position_usd=1_000_000,
        max_daily_loss_usd=1_000_000,
        max_orders_per_min=999,
    )
    exe = Executor(broker, risk)
    journal = MagicMock()
    journal.run_id = "test"

    from aitrade.strategy.examples.sma_crossover import SmaCrossover

    strat = SmaCrossover(symbol="AAPL", fast=5, slow=20)
    cfg = PaperRunConfig(
        symbol="AAPL",
        timeframe=Timeframe.MIN_1,
        duration=timedelta(seconds=0),
        poll_interval_secs=0,  # don't actually sleep in tests
        target_position_notional=2_000.0,
    )
    # Must return normally — not raise.
    run_paper_session(strat, broker, data, exe, journal, cfg)
    assert broker.submitted == []


def test_get_positions_failure_does_not_crash_session() -> None:
    """broker.get_positions raising mid-loop must not propagate."""
    live_start = datetime.now(UTC) - timedelta(minutes=2)
    df = _bars_df(live_start, n=60)

    broker = FakeBroker()
    # Make every get_positions raise.
    original = broker.get_positions
    broker.get_positions = MagicMock(side_effect=ConnectionError("net down"))  # type: ignore[method-assign]
    _ = original  # silence linter

    data = MagicMock()
    data.fetch_stock_bars.return_value = df

    risk = RiskGate(
        max_position_usd=1_000_000,
        max_daily_loss_usd=1_000_000,
        max_orders_per_min=999,
    )
    exe = Executor(broker, risk)
    journal = MagicMock()
    journal.run_id = "test"

    from aitrade.strategy.examples.sma_crossover import SmaCrossover

    strat = SmaCrossover(symbol="AAPL", fast=5, slow=20)
    cfg = PaperRunConfig(
        symbol="AAPL",
        timeframe=Timeframe.MIN_1,
        duration=timedelta(seconds=0),
        poll_interval_secs=0,
        target_position_notional=2_000.0,
    )
    # Should not raise. Position is treated as flat, so a LONG signal still
    # results in a buy attempt; that's fine.
    run_paper_session(strat, broker, data, exe, journal, cfg)
