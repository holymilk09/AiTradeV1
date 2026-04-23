"""Paper-trading bot runner.

Polls recent bars on a timeframe cadence, dispatches to the strategy, converts
signals into orders, runs them through the risk gate + executor, and logs
every event through the trade journal.

Designed for paper sessions. The same runner can power live sessions once the
broker is constructed with ``confirm_live=True``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from loguru import logger

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.brokers.base import BrokerClient
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Timeframe
from aitrade.execution.executor import Executor
from aitrade.execution.orders import OrderRequest, OrderType, Side, TimeInForce
from aitrade.logging.trade_logger import TradeLogger
from aitrade.strategy.base import Strategy
from aitrade.strategy.signal import Direction, size_by_notional


@dataclass
class PaperRunConfig:
    symbol: str
    timeframe: Timeframe
    duration: timedelta
    poll_interval_secs: int = 60
    target_position_notional: float = 2_000.0


def run_paper_session(
    strategy: Strategy,
    broker: BrokerClient,
    data: AlpacaDataClient,
    executor: Executor,
    journal: TradeLogger,
    cfg: PaperRunConfig,
) -> None:
    if not broker.is_paper:
        raise RuntimeError("paper runner refuses non-paper broker")

    end_at = datetime.now(UTC) + cfg.duration
    seen_ts: set[datetime] = set()
    last_direction: Direction = Direction.FLAT

    logger.info(
        "paper session start strategy={} symbol={} duration={}",
        strategy.strategy_id,
        cfg.symbol,
        cfg.duration,
    )
    while datetime.now(UTC) < end_at:
        start = datetime.now(UTC) - timedelta(days=5)
        df = data.fetch_stock_bars(
            cfg.symbol, cfg.timeframe, start=start, end=datetime.now(UTC), use_cache=False
        )
        if df.empty:
            logger.warning("no bars yet for {}", cfg.symbol)
            time.sleep(cfg.poll_interval_secs)
            continue

        for bar in df_to_bars(df, cfg.symbol):
            if bar.timestamp in seen_ts:
                continue
            seen_ts.add(bar.timestamp)
            signal = strategy.on_bar(bar)
            if signal is None or signal.direction == last_direction:
                continue
            last_direction = signal.direction

            positions = {p.symbol: p.qty for p in broker.get_positions()}
            held = positions.get(cfg.symbol, 0.0)

            if signal.direction is Direction.LONG:
                qty = size_by_notional(
                    notional_usd=cfg.target_position_notional, price=bar.close
                )
                delta = qty - held
                side = Side.BUY if delta > 0 else Side.SELL
                if abs(delta) < 1:
                    continue
                order = OrderRequest(
                    symbol=cfg.symbol,
                    side=side,
                    qty=abs(delta),
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=strategy.strategy_id,
                )
            elif signal.direction is Direction.FLAT and held > 0:
                order = OrderRequest(
                    symbol=cfg.symbol,
                    side=Side.SELL,
                    qty=held,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id=strategy.strategy_id,
                )
            else:
                continue

            journal.log_submit(order)
            result = executor.submit(
                order, reference_price=bar.close, current_position_qty=held
            )
            if not result.accepted:
                journal.log_risk_block(order, result.reason)
            elif result.ack is not None:
                journal.log_ack(order, result.ack)

        time.sleep(cfg.poll_interval_secs)

    logger.info("paper session done run_id={}", journal.run_id)
