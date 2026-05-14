"""Reasoner-driven paper/live session.

Poll bars on a cadence, compute indicators, hand the full context to the
LLM reasoner, route the reasoner's decision through the risk gate, submit
to Alpaca, and log every event — including the reasoner's `reason` — to
the trade journal.

The reasoner's rate limit bounds token spend independent of the poll rate.
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
from aitrade.logging.trade_logger import EventType, TradeLogger
from aitrade.reasoning.claude_reasoner import Reasoner
from aitrade.reasoning.decision import ReasonerInput
from aitrade.strategy.indicators import compute_snapshot
from aitrade.strategy.signal import Direction


@dataclass
class ReasonerRunConfig:
    symbol: str
    timeframe: Timeframe
    duration: timedelta
    poll_interval_secs: int = 60
    bar_history_days: int = 5


def _size_from_notional(notional_usd: float, price: float) -> float:
    if price <= 0:
        return 0.0
    qty = notional_usd // price
    return max(qty, 0.0)


def run_reasoner_session(
    reasoner: Reasoner,
    broker: BrokerClient,
    data: AlpacaDataClient,
    executor: Executor,
    journal: TradeLogger,
    cfg: ReasonerRunConfig,
) -> None:
    if not broker.is_paper:
        raise RuntimeError("reasoner runner refuses non-paper broker by default")

    end_at = datetime.now(UTC) + cfg.duration
    logger.info(
        "reasoner session start symbol={} tf={} duration={}",
        cfg.symbol,
        cfg.timeframe.value,
        cfg.duration,
    )

    while datetime.now(UTC) < end_at:
        start = datetime.now(UTC) - timedelta(days=cfg.bar_history_days)
        df = data.fetch_stock_bars(
            cfg.symbol,
            cfg.timeframe,
            start=start,
            end=datetime.now(UTC),
            use_cache=False,
        )
        if df.empty:
            logger.warning("no bars yet for {}", cfg.symbol)
            time.sleep(cfg.poll_interval_secs)
            continue

        bars = list(df_to_bars(df, cfg.symbol))
        snapshot = compute_snapshot(bars)

        account = broker.get_account()
        positions = {p.symbol: p for p in broker.get_positions()}
        pos = positions.get(cfg.symbol)
        held_qty = pos.qty if pos else 0.0
        avg_entry = pos.avg_entry_price if pos else None

        ctx = ReasonerInput(
            symbol=cfg.symbol,
            indicators=snapshot,
            current_position_qty=held_qty,
            avg_entry_price=avg_entry,
            cash_available=account.cash,
            buying_power=account.buying_power,
            max_position_usd=executor.risk.max_position_usd,
        )
        decision = reasoner.decide(ctx)
        if decision is None:
            # rate-limited or reasoner error — skip this cycle
            time.sleep(cfg.poll_interval_secs)
            continue

        journal.record(
            EventType.REASONER_DECISION,
            symbol=cfg.symbol,
            payload={"decision": decision.model_dump()},
        )

        if not decision.should_trade:
            logger.info("reasoner said hold symbol={}", cfg.symbol)
            time.sleep(cfg.poll_interval_secs)
            continue

        # Translate decision → order
        price = snapshot.price
        desired_qty = _size_from_notional(decision.target_notional_usd, price)

        if decision.direction is Direction.FLAT or desired_qty == 0:
            if held_qty > 0:
                order = OrderRequest(
                    symbol=cfg.symbol,
                    side=Side.SELL,
                    qty=held_qty,
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.DAY,
                    strategy_id="llm_advised",
                )
            else:
                time.sleep(cfg.poll_interval_secs)
                continue
        elif decision.direction is Direction.LONG:
            delta = desired_qty - held_qty
            if abs(delta) < 1:
                time.sleep(cfg.poll_interval_secs)
                continue
            order = OrderRequest(
                symbol=cfg.symbol,
                side=Side.BUY if delta > 0 else Side.SELL,
                qty=abs(delta),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                strategy_id="llm_advised",
            )
        else:
            # short not enabled in this MVP
            logger.info("reasoner requested short; not enabled, skipping")
            time.sleep(cfg.poll_interval_secs)
            continue

        journal.log_submit(order)
        result = executor.submit(
            order, reference_price=price, current_position_qty=held_qty
        )
        if not result.accepted:
            journal.log_risk_block(order, result.reason)
        elif result.ack is not None:
            journal.log_ack(order, result.ack)

        time.sleep(cfg.poll_interval_secs)

    logger.info("reasoner session done run_id={}", journal.run_id)
