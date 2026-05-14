"""Round-trip reconciler — FIFO, idempotency, pyramiding, PnL bucketing."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aitrade.execution.orders import Fill, Side
from aitrade.journal.round_trips import (
    ExitReason,
    PnlBucket,
    RoundTripReconciler,
)
from aitrade.logging.trade_logger import EventType, TradeLogger


def _log_fill(
    journal: TradeLogger,
    *,
    symbol: str,
    side: Side,
    qty: float,
    price: float,
    extra: dict[str, Any] | None = None,
) -> None:
    fill = Fill(
        client_order_id=f"co-{symbol}-{side.value}-{qty}",
        broker_order_id=f"br-{symbol}-{side.value}-{qty}",
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        filled_at=datetime.now(UTC),
    )
    payload: dict[str, Any] = {"fill": fill}
    if extra:
        payload.update(extra)
    journal.record(EventType.ORDER_FILLED, symbol=symbol, payload=payload)


def test_simple_round_trip(tmp_path: Path) -> None:
    """BUY 10 @ 100, SELL 10 @ 110 → one round trip with $100 PnL."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _log_fill(journal, symbol="AAPL", side=Side.BUY, qty=10, price=100.0)
        _log_fill(journal, symbol="AAPL", side=Side.SELL, qty=10, price=110.0)

        reconciler = RoundTripReconciler(journal)
        rts = reconciler.reconcile()

        assert len(rts) == 1
        rt = rts[0]
        assert rt.symbol == "AAPL"
        assert rt.qty == 10
        assert rt.entry_price == 100.0
        assert rt.exit_price == 110.0
        assert rt.pnl_usd == 100.0
        assert abs(rt.pnl_pct - 0.10) < 1e-9
        assert rt.pnl_bucket is PnlBucket.WIN
        assert rt.exit_reason is ExitReason.UNKNOWN


def test_pyramiding_fifo(tmp_path: Path) -> None:
    """BUY 10 @ 100, BUY 5 @ 110, SELL 8 @ 120 → first RT closes 8 from first lot."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _log_fill(journal, symbol="MSFT", side=Side.BUY, qty=10, price=100.0)
        _log_fill(journal, symbol="MSFT", side=Side.BUY, qty=5, price=110.0)
        _log_fill(journal, symbol="MSFT", side=Side.SELL, qty=8, price=120.0)

        reconciler = RoundTripReconciler(journal)
        rts = reconciler.reconcile()

        assert len(rts) == 1
        rt = rts[0]
        assert rt.qty == 8
        assert rt.entry_price == 100.0  # came from FIRST lot
        assert rt.exit_price == 120.0
        assert rt.pnl_usd == 8 * 20.0  # 8 shares * $20 gain
        assert rt.pnl_bucket is PnlBucket.WIN

        # Open positions should still hold 2 from lot1 + 5 from lot2.
        cur = journal._conn.execute(  # noqa: SLF001 — inspecting test fixture
            "SELECT entry_price, qty_remaining FROM open_positions ORDER BY entry_ts"
        )
        rows = cur.fetchall()
        assert len(rows) == 2
        assert (rows[0][0], rows[0][1]) == (100.0, 2.0)
        assert (rows[1][0], rows[1][1]) == (110.0, 5.0)


def test_pnl_bucket_boundaries(tmp_path: Path) -> None:
    """WIN above +0.1%, LOSS below -0.1%, BREAKEVEN inside the band."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # Loss
        _log_fill(journal, symbol="LOSE", side=Side.BUY, qty=1, price=100.0)
        _log_fill(journal, symbol="LOSE", side=Side.SELL, qty=1, price=98.0)
        # Breakeven (within 0.1%)
        _log_fill(journal, symbol="EVEN", side=Side.BUY, qty=1, price=100.0)
        _log_fill(journal, symbol="EVEN", side=Side.SELL, qty=1, price=100.05)
        # Win
        _log_fill(journal, symbol="WIN", side=Side.BUY, qty=1, price=100.0)
        _log_fill(journal, symbol="WIN", side=Side.SELL, qty=1, price=101.0)

        reconciler = RoundTripReconciler(journal)
        rts = reconciler.reconcile()

        by_symbol = {rt.symbol: rt for rt in rts}
        assert by_symbol["LOSE"].pnl_bucket is PnlBucket.LOSS
        assert by_symbol["EVEN"].pnl_bucket is PnlBucket.BREAKEVEN
        assert by_symbol["WIN"].pnl_bucket is PnlBucket.WIN


def test_reconciler_is_idempotent(tmp_path: Path) -> None:
    """Calling reconcile twice over the same events yields no duplicates."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _log_fill(journal, symbol="NVDA", side=Side.BUY, qty=4, price=500.0)
        _log_fill(journal, symbol="NVDA", side=Side.SELL, qty=4, price=510.0)

        reconciler = RoundTripReconciler(journal)
        first = reconciler.reconcile()
        second = reconciler.reconcile()

        assert len(first) == 1
        assert second == []  # nothing new
        # And the persisted table still has exactly one row.
        all_rts = reconciler.all_round_trips()
        assert len(all_rts) == 1
        assert all_rts[0].symbol == "NVDA"


def test_partial_close_then_flat(tmp_path: Path) -> None:
    """BUY 10, SELL 4, SELL 6 → two round trips emitted in time order."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _log_fill(journal, symbol="TSLA", side=Side.BUY, qty=10, price=200.0)
        _log_fill(
            journal,
            symbol="TSLA",
            side=Side.SELL,
            qty=4,
            price=210.0,
            extra={"exit_reason": ExitReason.TARGET_HIT.value},
        )
        _log_fill(
            journal,
            symbol="TSLA",
            side=Side.SELL,
            qty=6,
            price=205.0,
            extra={"exit_reason": ExitReason.MANUAL_EXIT.value},
        )

        reconciler = RoundTripReconciler(journal)
        rts = reconciler.reconcile()

        assert len(rts) == 2
        first, second = rts
        assert first.qty == 4
        assert first.exit_price == 210.0
        assert first.exit_reason is ExitReason.TARGET_HIT
        assert second.qty == 6
        assert second.exit_price == 205.0
        assert second.exit_reason is ExitReason.MANUAL_EXIT
        # No open lots remain.
        cur = journal._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM open_positions"
        )
        assert cur.fetchone()[0] == 0
