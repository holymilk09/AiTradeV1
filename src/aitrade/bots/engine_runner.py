"""Phase 1 engine runner — discover → patterns → board → reasoner → execute.

End-to-end orchestration of the signal engine. Each cycle:

  1. ``DiscoveryAgent.discover()`` — Claude + web_search surfaces buzzy tickers.
  2. ``MarketSnapshotFetcher.fetch()`` — SPY/QQQ/VIX with strict freshness.
  3. For each candidate:
       a. fetch multi-TF bars (1D / 1H / 5m), no cache (live data only)
       b. ``assert_bars_fresh`` against per-TF threshold — raises if stale
       c. compute ``MultiTimeframeSnapshot`` + run all pattern detectors
  4. Build a ranked ``CandidateBoard`` (combined buzz + pattern z-scores).
  5. ``FloorTraderReasoner.decide_board`` — Claude picks ONE or passes.
  6. Translate to ``OrderRequest`` → ``RiskGate`` → ``Executor`` → Alpaca.
  7. ``RoundTripReconciler.reconcile`` after every cycle so closed positions
     emit ``ROUND_TRIP_CLOSED`` events for Phase 1.5 experience replay.

The whole loop is journaled — every discovery scan, board, decision, order,
and round-trip lands in ``logs/trades.jsonl`` + the SQLite mirror.

The engine never panics. Any exception in a single cycle is logged + skipped;
the next tick proceeds. ``KILL_SWITCH`` file in the cwd halts new orders via
the existing ``RiskGate``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from loguru import logger

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.brokers.base import BrokerClient
from aitrade.calendar.client import EconomicCalendarClient
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Bar, Timeframe
from aitrade.discovery.agent import DiscoveryAgent
from aitrade.discovery.movers import MoversFinder
from aitrade.discovery.reddit import RedditDiscoveryClient
from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.discovery.universe import build_universe
from aitrade.execution.executor import Executor
from aitrade.execution.orders import OrderRequest, OrderType, Side, TimeInForce
from aitrade.journal.narratives import NarrativeGenerator
from aitrade.journal.round_trips import RoundTripReconciler, TradeRoundTrip
from aitrade.journal.similarity import SimilarityQuery, SimilarTradesFinder
from aitrade.logging.trade_logger import EventType, TradeLogger
from aitrade.market.freshness import StaleDataError, assert_bars_fresh
from aitrade.market.snapshot import MarketSnapshotFetcher
from aitrade.news.client import AlpacaNewsClient
from aitrade.patterns.base import PatternSignal
from aitrade.patterns.board import CandidateBoard, build_board
from aitrade.patterns.registry import get_all_detectors
from aitrade.reasoning.decision import FloorTraderDecision
from aitrade.reasoning.floor_trader import FloorTraderInput, FloorTraderReasoner
from aitrade.strategy.indicators import compute_multi_tf_snapshot
from aitrade.strategy.signal import Direction

# Timeframes scanned per candidate. Daily for macro, 1H for structure,
# 5m for entry timing. Bar counts are sized so every indicator (incl.
# SMA50, SMA200, MACD-9-of-26) has enough lookback at each resolution.
_TIMEFRAMES: dict[Timeframe, int] = {
    Timeframe.DAY_1: 200,
    Timeframe.HOUR_1: 60,
    Timeframe.MIN_5: 60,
}


@dataclass
class EngineConfig:
    """Per-run settings for the engine. Defaults read from ``Settings`` env."""

    cycle_secs: int = 900
    discovery_top_n: int = 20
    target_notional_per_trade: float = 2_000.0
    max_cycles: int | None = None
    duration: timedelta | None = None
    # Universe sources — deterministic + cheap on, social/web layers opt-in.
    use_watchlist: bool = True
    use_movers: bool = True
    use_reddit_discovery: bool = False
    use_web_discovery: bool = False
    # Phase 2 — news + calendar enrichment for the floor-trader prompt.
    news_lookback_hours: int = 24
    news_per_symbol: int = 3
    calendar_days_ahead: int = 7


def _bars_lookback_window(tf: Timeframe, count: int) -> tuple[datetime, datetime]:
    """Coarse but generous start/end for fetching ``count`` bars at ``tf``."""
    end = datetime.now(UTC)
    span = {
        Timeframe.MIN_1: timedelta(days=2),
        Timeframe.MIN_5: timedelta(days=5),
        Timeframe.MIN_15: timedelta(days=10),
        Timeframe.HOUR_1: timedelta(days=20),
        Timeframe.DAY_1: timedelta(days=400),
    }[tf]
    return end - span, end


def _fetch_multi_tf_bars(
    data: AlpacaDataClient,
    symbol: str,
) -> dict[Timeframe, list[Bar]]:
    """Pull each timeframe live (use_cache=False), enforce freshness per TF.

    Raises ``StaleDataError`` if any timeframe is empty or its last bar is
    older than the per-TF threshold (see ``aitrade.market.freshness``).
    """
    out: dict[Timeframe, list[Bar]] = {}
    for tf, count in _TIMEFRAMES.items():
        start, end = _bars_lookback_window(tf, count)
        df = data.fetch_stock_bars(symbol, tf, start=start, end=end, use_cache=False)
        bars = list(df_to_bars(df, symbol)) if not df.empty else []
        assert_bars_fresh(bars, tf)
        # Trim to the most recent ``count`` bars to bound indicator work.
        out[tf] = bars[-count:]
    return out


def _scan_patterns(symbol: str, daily_bars: list[Bar]) -> list[PatternSignal]:
    """Run all registered detectors on the daily bars; collect non-None hits."""
    hits: list[PatternSignal] = []
    for detector in get_all_detectors():
        try:
            sig = detector.detect(daily_bars)
        except Exception as e:  # pragma: no cover — detectors must not raise
            logger.warning("detector {} raised on {}: {}", detector.name, symbol, e)
            continue
        if sig is not None:
            hits.append(sig)
    return hits


def _compact_round_trip(rt: TradeRoundTrip) -> dict[str, Any]:
    """Compact a closed round-trip into the prompt-friendly shape the
    floor-trader's ``prior_experience`` field expects.

    Keep it short — three to five short sentences worth of JSON per row,
    so a top-3 retrieval fits comfortably in the reasoner's context.
    """
    snapshot: dict[str, Any] = rt.market_snapshot or {}
    regime = snapshot.get("regime") if isinstance(snapshot, dict) else None
    held_h = max(1, rt.holding_secs // 3600)
    return {
        "when": rt.entry_ts.strftime("%Y-%m-%d"),
        "pattern": ", ".join(rt.pattern_hits) if rt.pattern_hits else "n/a",
        "outcome": f"{rt.pnl_bucket.value} {rt.pnl_pct * 100:+.2f}%",
        "thesis": rt.entry_thesis or "",
        "what_happened": (
            f"Bought {rt.qty:g} @ {rt.entry_price:.2f}, "
            f"sold @ {rt.exit_price:.2f} ({rt.exit_reason.value}, ~{held_h}h held)"
        ),
        "market_then": regime,
    }


def _build_prior_experience(
    board: CandidateBoard,
    snapshot_regime: str | None,
    finder: SimilarTradesFinder | None,
    *,
    consider_top_k_candidates: int = 5,
    per_symbol_top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """For the top-K board candidates, fetch up to ``per_symbol_top_k`` similar
    past round-trips and project each through ``_compact_round_trip``.

    Empty dict if the finder is None or every candidate yields no matches —
    that's the correct signal for the floor-trader: "no prior data".
    """
    if finder is None:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for cand in board.top(consider_top_k_candidates):
        query = SimilarityQuery(
            symbol=cand.symbol,
            pattern_hits=list(cand.pattern_hits),
            market_regime=snapshot_regime,
        )
        try:
            similar = finder.find(query, top_k=per_symbol_top_k)
        except Exception as e:  # pragma: no cover — finder is pure SQL+Python
            logger.warning("similarity find failed for {}: {}", cand.symbol, e)
            continue
        if similar:
            out[cand.symbol] = [_compact_round_trip(rt) for rt in similar]
    return out


def _build_news_dump(
    board: CandidateBoard,
    news_client: AlpacaNewsClient | None,
    *,
    consider_top_k: int = 5,
    lookback_hours: int = 24,
    per_symbol: int = 3,
) -> dict[str, list[dict[str, object]]]:
    """For the top-K board candidates, fetch recent headlines and project them
    through ``NewsItem.to_compact``. Empty dict if news_client is None or the
    fetch returns nothing — a benign signal to the floor-trader.
    """
    if news_client is None:
        return {}
    symbols = [c.symbol for c in board.top(consider_top_k)]
    if not symbols:
        return {}
    try:
        raw = news_client.fetch(
            symbols, lookback_hours=lookback_hours, limit_per_symbol=per_symbol
        )
    except Exception as e:  # pragma: no cover — client already swallows
        logger.warning("news fetch failed: {}", e)
        return {}
    return {
        sym: [cast("dict[str, object]", item.to_compact()) for item in items]
        for sym, items in raw.items()
    }


def _build_calendar_dump(
    cal_client: EconomicCalendarClient | None,
    candidate_symbols: list[str],
    *,
    days_ahead: int = 7,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return (economic_calendar, upcoming_earnings) projections for the prompt.
    Returns ([], []) if cal_client is None or fetches fail."""
    if cal_client is None:
        return [], []
    econ: list[dict[str, object]] = []
    earn: list[dict[str, object]] = []
    try:
        econ = [
            cast("dict[str, object]", e.to_compact())
            for e in cal_client.economic_events(days_ahead=days_ahead)
        ]
    except Exception as e:  # pragma: no cover — client already swallows
        logger.warning("economic calendar fetch failed: {}", e)
    try:
        earn = [
            cast("dict[str, object]", e.to_compact())
            for e in cal_client.earnings_events(
                days_ahead=days_ahead,
                symbols=candidate_symbols or None,
            )
        ]
    except Exception as e:  # pragma: no cover — client already swallows
        logger.warning("earnings calendar fetch failed: {}", e)
    return econ, earn


def _build_floor_trader_input(  # noqa: PLR0913 — explicit context fan-in
    board: CandidateBoard,
    *,
    positions: dict[str, dict[str, Any]],
    cash_available: float,
    buying_power: float,
    max_position_usd: float,
    market_snapshot_dump: dict[str, Any],
    multi_tf_dumps: dict[str, dict[str, Any]],
    prior_experience: dict[str, list[dict[str, Any]]] | None = None,
    news_by_symbol: dict[str, list[dict[str, Any]]] | None = None,
    economic_calendar: list[dict[str, Any]] | None = None,
    upcoming_earnings: list[dict[str, Any]] | None = None,
) -> FloorTraderInput:
    return FloorTraderInput(
        candidate_board=[c.to_dict() for c in board.top(10)],
        current_positions=positions,
        cash_available=cash_available,
        buying_power=buying_power,
        max_position_usd=max_position_usd,
        market_snapshot=market_snapshot_dump,
        multi_tf_snapshots=multi_tf_dumps,
        prior_experience=prior_experience or {},
        news_by_symbol=news_by_symbol or {},
        economic_calendar=economic_calendar or [],
        upcoming_earnings=upcoming_earnings or [],
    )


def _decision_to_order(
    decision: FloorTraderDecision,
    *,
    held_qty: float,
    reference_price: float,
    target_notional: float,
) -> OrderRequest | None:
    """Translate a floor-trader decision into an Alpaca order, or None."""
    if not decision.should_trade or decision.pick_symbol is None:
        return None
    if reference_price <= 0:
        logger.warning("non-positive reference price for {}", decision.pick_symbol)
        return None

    if decision.direction is Direction.FLAT:
        if held_qty <= 0:
            return None
        return OrderRequest(
            symbol=decision.pick_symbol,
            side=Side.SELL,
            qty=held_qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="floor_trader",
        )

    if decision.direction is Direction.LONG:
        notional = min(decision.target_notional_usd or target_notional, target_notional)
        desired_qty = max(notional // reference_price, 1.0)
        delta = desired_qty - held_qty
        if abs(delta) < 1:
            return None
        return OrderRequest(
            symbol=decision.pick_symbol,
            side=Side.BUY if delta > 0 else Side.SELL,
            qty=abs(delta),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="floor_trader",
        )

    # SHORT explicitly unsupported in v1.
    logger.info("floor trader requested {} for {}; v1 is long-only — skipping",
                decision.direction.value, decision.pick_symbol)
    return None


def _run_one_cycle(  # noqa: PLR0913 — orchestration glue, intentional fan-in
    *,
    discovery: DiscoveryAgent | None,
    movers: MoversFinder | None,
    reddit: RedditDiscoveryClient | None,
    news_client: AlpacaNewsClient | None,
    cal_client: EconomicCalendarClient | None,
    market_fetcher: MarketSnapshotFetcher,
    data: AlpacaDataClient,
    broker: BrokerClient,
    executor: Executor,
    reasoner: FloorTraderReasoner,
    similar_finder: SimilarTradesFinder | None,
    narrative_gen: NarrativeGenerator | None,
    journal: TradeLogger,
    reconciler: RoundTripReconciler,
    cfg: EngineConfig,
) -> None:
    cycle_start = datetime.now(UTC)
    logger.info("engine cycle start at {}", cycle_start.isoformat())

    discovered: list[DiscoveredTicker] = build_universe(
        movers=movers,
        reddit=reddit,
        web_agent=discovery,
        web_top_n=cfg.discovery_top_n,
        use_watchlist=cfg.use_watchlist,
        use_movers=cfg.use_movers,
        use_reddit=cfg.use_reddit_discovery,
        use_web=cfg.use_web_discovery,
    )
    # Cap to top-N so a hot day doesn't blow up the per-cycle bar fetches.
    discovered = discovered[: cfg.discovery_top_n]
    journal.record(
        EventType.DISCOVERY_SCAN,
        symbol="-",
        payload={
            "count": len(discovered),
            "use_watchlist": cfg.use_watchlist,
            "use_movers": cfg.use_movers,
            "use_reddit": cfg.use_reddit_discovery,
            "use_web": cfg.use_web_discovery,
            "tickers": [
                {"symbol": d.symbol, "buzz_score": d.buzz_score} for d in discovered
            ],
        },
    )
    if not discovered:
        journal.record(EventType.EMPTY_DISCOVERY, symbol="-", payload={})
        logger.info("universe is empty this cycle; skipping")
        return

    # Market snapshot first — bail if stale.
    try:
        snapshot = market_fetcher.fetch()
    except StaleDataError as e:
        journal.record(
            EventType.STALE_DATA_SKIP,
            symbol="-",
            payload={"reason": "market_snapshot", "error": str(e)},
        )
        logger.warning("stale market snapshot; skipping cycle: {}", e)
        return

    snapshot_dump: dict[str, Any] = {
        "spy_price": snapshot.spy_price,
        "spy_change_pct": snapshot.spy_change_pct,
        "qqq_price": snapshot.qqq_price,
        "qqq_change_pct": snapshot.qqq_change_pct,
        "vix": snapshot.vix,
        "regime": snapshot.regime.value,
        "session": snapshot.session,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "age_secs": round(snapshot.age_secs(), 2),
    }
    journal.record(EventType.MARKET_SNAPSHOT, symbol="-", payload=snapshot_dump)

    # Per-symbol scan: bars → MTF → patterns.
    patterns_by_symbol: dict[str, list[PatternSignal]] = {}
    last_price_by_symbol: dict[str, float] = {}
    multi_tf_dumps: dict[str, dict[str, Any]] = {}

    for ticker in discovered:
        sym = ticker.symbol
        try:
            bars_by_tf = _fetch_multi_tf_bars(data, sym)
        except StaleDataError as e:
            logger.warning("stale bars for {}; dropping from cycle: {}", sym, e)
            continue
        except Exception as e:  # pragma: no cover — alpaca-side noise
            logger.warning("bar fetch failed for {}: {}", sym, e)
            continue

        daily_bars = bars_by_tf.get(Timeframe.DAY_1, [])
        if not daily_bars:
            continue
        last_price_by_symbol[sym] = daily_bars[-1].close

        try:
            mtf = compute_multi_tf_snapshot(bars_by_tf)
        except ValueError as e:
            logger.warning("MTF snapshot failed for {}: {}", sym, e)
            continue
        multi_tf_dumps[sym] = mtf.as_dict()

        hits = _scan_patterns(sym, daily_bars)
        if hits:
            patterns_by_symbol[sym] = hits
            journal.record(
                EventType.PATTERN_DETECTED,
                symbol=sym,
                payload={
                    "patterns": [
                        {
                            "name": h.name,
                            "score": h.score,
                            "direction": h.direction.value,
                            "evidence": h.evidence,
                        }
                        for h in hits
                    ]
                },
            )

    # Account state.
    account = broker.get_account()
    positions = broker.get_positions()
    held_qty_by_symbol = {p.symbol: p.qty for p in positions}
    positions_dump: dict[str, dict[str, Any]] = {
        p.symbol: {
            "qty": p.qty,
            "avg_entry_price": p.avg_entry_price,
            "market_value": p.market_value,
            "unrealized_pnl": p.unrealized_pnl,
        }
        for p in positions
    }

    board = build_board(
        discovered,
        patterns_by_symbol,
        held_qty_by_symbol=held_qty_by_symbol,
        last_price_by_symbol=last_price_by_symbol,
        cash_available=account.cash,
        target_notional=cfg.target_notional_per_trade,
        now=cycle_start,
    )
    journal.record(
        EventType.CANDIDATE_BOARD,
        symbol="-",
        payload=board.to_dict(),
    )
    if not board.candidates:
        logger.info("empty candidate board; skipping cycle")
        return

    prior_experience = _build_prior_experience(
        board, snapshot.regime.value, similar_finder
    )
    news_dump = _build_news_dump(
        board,
        news_client,
        lookback_hours=cfg.news_lookback_hours,
        per_symbol=cfg.news_per_symbol,
    )
    candidate_symbols = [c.symbol for c in board.top(10)]
    econ_events, earn_events = _build_calendar_dump(
        cal_client, candidate_symbols, days_ahead=cfg.calendar_days_ahead
    )
    decision = reasoner.decide_board(
        _build_floor_trader_input(
            board,
            positions=positions_dump,
            cash_available=account.cash,
            buying_power=account.buying_power,
            max_position_usd=executor.risk.max_position_usd,
            market_snapshot_dump=snapshot_dump,
            multi_tf_dumps=multi_tf_dumps,
            prior_experience=prior_experience,
            news_by_symbol=news_dump,
            economic_calendar=econ_events,
            upcoming_earnings=earn_events,
        )
    )
    if decision is None:
        logger.info("reasoner returned None (rate-limited or error); skipping cycle")
        return

    journal.record(
        EventType.FLOOR_TRADER_DECISION,
        symbol=decision.pick_symbol or "-",
        payload=decision.model_dump(mode="json"),
    )

    if not decision.should_trade or decision.pick_symbol is None:
        logger.info(
            "floor trader passed: {}",
            decision.reason_for_pass or "no reason given",
        )
        return

    sym = decision.pick_symbol
    held_qty = held_qty_by_symbol.get(sym, 0.0)
    reference_price = last_price_by_symbol.get(sym, 0.0)
    if reference_price <= 0:
        logger.warning("no reference price for picked symbol {}", sym)
        return

    order = _decision_to_order(
        decision,
        held_qty=held_qty,
        reference_price=reference_price,
        target_notional=cfg.target_notional_per_trade,
    )
    if order is None:
        logger.info("no order needed (already at target or unsupported direction)")
        return

    journal.log_submit(order)
    result = executor.submit(
        order, reference_price=reference_price, current_position_qty=held_qty
    )
    if not result.accepted:
        journal.log_risk_block(order, result.reason)
        return
    if result.ack is not None:
        journal.log_ack(order, result.ack)

    # Reconcile after each cycle so newly closed round-trips land promptly.
    new_round_trips = reconciler.reconcile()
    for rt in new_round_trips:
        journal.record(
            EventType.ROUND_TRIP_CLOSED,
            symbol=rt.symbol,
            payload={
                "trade_id": rt.trade_id,
                "pnl_usd": rt.pnl_usd,
                "pnl_pct": rt.pnl_pct,
                "pnl_bucket": rt.pnl_bucket.value,
                "exit_reason": rt.exit_reason.value,
                "holding_secs": rt.holding_secs,
                "qty": rt.qty,
                "entry_price": rt.entry_price,
                "exit_price": rt.exit_price,
            },
        )
        # Phase 1.5: write a Claude-Haiku post-mortem narrative for the closed
        # round-trip. Synchronous (~1s/call); idempotent and best-effort, never
        # raises. Result is queryable via JournalViews + similarity retrieval.
        if narrative_gen is not None:
            narrative_gen.write_for(rt)


def run_engine(  # noqa: PLR0913 — top-level orchestrator, intentional fan-in
    *,
    discovery: DiscoveryAgent | None,
    movers: MoversFinder | None,
    reddit: RedditDiscoveryClient | None,
    news_client: AlpacaNewsClient | None,
    cal_client: EconomicCalendarClient | None,
    market_fetcher: MarketSnapshotFetcher,
    data: AlpacaDataClient,
    broker: BrokerClient,
    executor: Executor,
    reasoner: FloorTraderReasoner,
    similar_finder: SimilarTradesFinder | None,
    narrative_gen: NarrativeGenerator | None,
    journal: TradeLogger,
    reconciler: RoundTripReconciler,
    cfg: EngineConfig,
) -> None:
    """Top-level engine loop. Refuses non-paper brokers by default."""
    if not broker.is_paper:
        raise RuntimeError("engine refuses non-paper broker by default")

    end_at: datetime | None = None
    if cfg.duration is not None:
        end_at = datetime.now(UTC) + cfg.duration

    cycle = 0
    while True:
        try:
            _run_one_cycle(
                discovery=discovery,
                movers=movers,
                reddit=reddit,
                news_client=news_client,
                cal_client=cal_client,
                market_fetcher=market_fetcher,
                data=data,
                broker=broker,
                executor=executor,
                reasoner=reasoner,
                similar_finder=similar_finder,
                narrative_gen=narrative_gen,
                journal=journal,
                reconciler=reconciler,
                cfg=cfg,
            )
        except Exception as e:  # pragma: no cover — engine never panics
            logger.exception("engine cycle raised; continuing: {}", e)

        cycle += 1
        if cfg.max_cycles is not None and cycle >= cfg.max_cycles:
            logger.info("engine reached max_cycles={}; exiting", cfg.max_cycles)
            return
        if end_at is not None and datetime.now(UTC) >= end_at:
            logger.info("engine reached duration limit; exiting")
            return

        time.sleep(cfg.cycle_secs)
