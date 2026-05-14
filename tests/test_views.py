"""SQL-backed journal views — filters, pattern stats, LLM digest."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aitrade.journal.round_trips import (
    ExitReason,
    PnlBucket,
    RoundTripReconciler,
)
from aitrade.journal.views import JournalViews, PatternStats, SummaryStats
from aitrade.logging.trade_logger import TradeLogger


def _seed(
    journal: TradeLogger,
    *,
    trade_id: str,
    symbol: str,
    pattern_hits: list[str],
    bucket: PnlBucket,
    pnl_usd: float,
    pnl_pct: float,
    exit_ts: datetime,
    entry_thesis: str | None = None,
    regime: str | None = None,
    exit_reason: ExitReason = ExitReason.UNKNOWN,
) -> None:
    """Insert a synthetic round-trip row."""
    RoundTripReconciler(journal)  # ensure schema
    entry_ts = exit_ts - timedelta(hours=1)
    market_snapshot = json.dumps({"regime": regime}) if regime else None
    journal._conn.execute(  # noqa: SLF001
        "INSERT INTO round_trips VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trade_id,
            symbol,
            "test-strat",
            f"ev-entry-{trade_id}",
            f"ev-exit-{trade_id}",
            entry_ts.isoformat(),
            exit_ts.isoformat(),
            100.0,
            100.0 * (1 + pnl_pct),
            1.0,
            pnl_usd,
            pnl_pct,
            3600,
            bucket.value,
            exit_reason.value,
            entry_thesis,
            None,
            None,
            json.dumps(pattern_hits),
            market_snapshot,
        ),
    )


# --- round_trips ------------------------------------------------------------


def test_round_trips_filters_by_symbol(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="a1", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01, exit_ts=base)
        _seed(journal, trade_id="m1", symbol="MSFT", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
              exit_ts=base + timedelta(hours=1))

        views = JournalViews(journal)
        only_aapl = views.round_trips(symbol="AAPL")
        assert [rt.trade_id for rt in only_aapl] == ["a1"]


def test_round_trips_filter_by_pattern_string_match(tmp_path: Path) -> None:
    """Pattern filter must match exact name, not substring of another name."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="hit", symbol="AAPL", pattern_hits=["breakout"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01, exit_ts=base)
        _seed(journal, trade_id="miss", symbol="MSFT", pattern_hits=["volume_trend"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
              exit_ts=base + timedelta(hours=1))
        # Multi-pattern row that shares one name.
        _seed(journal, trade_id="multi", symbol="GOOG",
              pattern_hits=["breakout", "consolidation"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
              exit_ts=base + timedelta(hours=2))

        views = JournalViews(journal)
        result = views.round_trips(pattern="breakout")
        ids = sorted(rt.trade_id for rt in result)
        assert ids == ["hit", "multi"]


def test_round_trips_pattern_avoids_substring_false_positive(tmp_path: Path) -> None:
    """Quoted-substring LIKE must not match a pattern that's a prefix of another."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="vol-trend", symbol="AAPL",
              pattern_hits=["volume_trend"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01, exit_ts=base)
        views = JournalViews(journal)
        # Searching for "vol" should not match "volume_trend".
        assert views.round_trips(pattern="vol") == []
        # Exact match works.
        result = views.round_trips(pattern="volume_trend")
        assert [rt.trade_id for rt in result] == ["vol-trend"]


def test_round_trips_wins_only_and_losses_only(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="w", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01, exit_ts=base)
        _seed(journal, trade_id="l", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.LOSS, pnl_usd=-1.0, pnl_pct=-0.01,
              exit_ts=base + timedelta(hours=1))
        _seed(journal, trade_id="b", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.BREAKEVEN, pnl_usd=0.0, pnl_pct=0.0,
              exit_ts=base + timedelta(hours=2))

        views = JournalViews(journal)
        wins = views.round_trips(wins_only=True)
        losses = views.round_trips(losses_only=True)
        assert [rt.trade_id for rt in wins] == ["w"]
        assert [rt.trade_id for rt in losses] == ["l"]
        # Mutually exclusive: passing both yields empty.
        assert views.round_trips(wins_only=True, losses_only=True) == []


def test_round_trips_orders_newest_first_and_caps_limit(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i in range(5):
            _seed(journal, trade_id=f"t-{i}", symbol="AAPL", pattern_hits=["x"],
                  bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
                  exit_ts=base + timedelta(hours=i))
        views = JournalViews(journal)
        result = views.round_trips(limit=2)
        assert [rt.trade_id for rt in result] == ["t-4", "t-3"]


# --- pattern_stats ----------------------------------------------------------


def test_pattern_stats_groups_and_sums(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # breakout: 2 wins, 1 loss; volume: 1 win.
        _seed(journal, trade_id="b1", symbol="AAPL",
              pattern_hits=["breakout", "volume"],
              bucket=PnlBucket.WIN, pnl_usd=10.0, pnl_pct=0.05, exit_ts=base)
        _seed(journal, trade_id="b2", symbol="MSFT",
              pattern_hits=["breakout"],
              bucket=PnlBucket.WIN, pnl_usd=20.0, pnl_pct=0.10,
              exit_ts=base + timedelta(hours=1))
        _seed(journal, trade_id="b3", symbol="GOOG",
              pattern_hits=["breakout"],
              bucket=PnlBucket.LOSS, pnl_usd=-5.0, pnl_pct=-0.025,
              exit_ts=base + timedelta(hours=2))

        views = JournalViews(journal)
        stats = {s.pattern: s for s in views.pattern_stats()}

        breakout = stats["breakout"]
        assert breakout.n_trades == 3
        assert breakout.n_wins == 2
        assert breakout.n_losses == 1
        assert breakout.n_breakeven == 0
        assert abs(breakout.win_rate - 2 / 3) < 1e-9
        assert abs(breakout.total_pnl_usd - 25.0) < 1e-9
        assert abs(breakout.avg_pnl_pct - (0.05 + 0.10 - 0.025) / 3) < 1e-9

        vol = stats["volume"]
        assert vol.n_trades == 1
        assert vol.n_wins == 1
        assert vol.win_rate == 1.0


def test_pattern_stats_breakeven_and_empty(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # Empty journal: zero stats.
        RoundTripReconciler(journal)
        views = JournalViews(journal)
        assert views.pattern_stats() == []

        _seed(journal, trade_id="be", symbol="AAPL",
              pattern_hits=["chop"],
              bucket=PnlBucket.BREAKEVEN, pnl_usd=0.0, pnl_pct=0.0, exit_ts=base)
        result = views.pattern_stats()
        assert len(result) == 1
        s = result[0]
        assert s.pattern == "chop"
        assert s.n_breakeven == 1
        assert s.win_rate == 0.0


def test_pattern_stats_win_rate_property() -> None:
    """The win_rate property is derived; doubles as a docs check."""
    s = PatternStats(
        pattern="x",
        n_trades=4,
        n_wins=3,
        n_losses=1,
        n_breakeven=0,
        avg_pnl_pct=0.01,
        total_pnl_usd=10.0,
    )
    assert s.win_rate == 0.75
    s_zero = PatternStats(
        pattern="y",
        n_trades=0,
        n_wins=0,
        n_losses=0,
        n_breakeven=0,
        avg_pnl_pct=0.0,
        total_pnl_usd=0.0,
    )
    assert s_zero.win_rate == 0.0


# --- export_for_llm ---------------------------------------------------------


def test_export_for_llm_works_without_narratives_table(tmp_path: Path) -> None:
    base = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="t-1", symbol="AAPL",
              pattern_hits=["breakout", "volume_trend"],
              bucket=PnlBucket.WIN, pnl_usd=3.30, pnl_pct=0.0191,
              entry_thesis="momentum + volume confirms",
              regime="risk_on_low_vol",
              exit_ts=base + timedelta(days=2),
              exit_reason=ExitReason.TARGET_HIT)

        # Sanity: trade_narratives table should not exist yet.
        cur = journal._conn.execute(  # noqa: SLF001
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_narratives'"
        )
        assert cur.fetchone() is None

        views = JournalViews(journal)
        digest = views.export_for_llm(max_trades=10)
        assert digest.startswith("# Trade journal — last 1 round-trips")
        assert "## TRADE-001 — AAPL — WIN +1.91%" in digest
        assert "Patterns: breakout, volume_trend" in digest
        assert "Regime at entry: risk_on_low_vol" in digest
        assert 'Thesis: "momentum + volume confirms"' in digest
        # No Narrative line when the table is absent.
        assert "Narrative:" not in digest
        assert "exit reason: TARGET_HIT" in digest


def test_export_for_llm_empty_journal(tmp_path: Path) -> None:
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        RoundTripReconciler(journal)
        views = JournalViews(journal)
        digest = views.export_for_llm()
        assert digest.startswith("# Trade journal — last 0 round-trips")
        assert "_No round-trips recorded yet._" in digest


def test_export_for_llm_with_narratives_table(tmp_path: Path) -> None:
    base = datetime(2026, 4, 20, 14, 30, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="t-1", symbol="AAPL", pattern_hits=["breakout"],
              bucket=PnlBucket.LOSS, pnl_usd=-2.0, pnl_pct=-0.0125,
              exit_ts=base, exit_reason=ExitReason.STOP_HIT)

        # Manually create the narratives table + a row for this trade.
        journal._conn.executescript(  # noqa: SLF001
            "CREATE TABLE IF NOT EXISTS trade_narratives ("
            "  trade_id TEXT PRIMARY KEY,"
            "  narrative TEXT NOT NULL,"
            "  created_at TEXT NOT NULL,"
            "  model TEXT NOT NULL"
            ");"
        )
        journal._conn.execute(  # noqa: SLF001
            "INSERT INTO trade_narratives VALUES (?, ?, ?, ?)",
            ("t-1", "Stop fired before pattern played out.",
             datetime.now(UTC).isoformat(), "claude-haiku"),
        )

        views = JournalViews(journal)
        digest = views.export_for_llm()
        assert "## TRADE-001 — AAPL — LOSS -1.25%" in digest
        assert 'Narrative: "Stop fired before pattern played out."' in digest


def test_export_for_llm_respects_max_trades(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i in range(5):
            _seed(journal, trade_id=f"t-{i}", symbol="AAPL", pattern_hits=["x"],
                  bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
                  exit_ts=base + timedelta(hours=i))
        views = JournalViews(journal)
        digest = views.export_for_llm(max_trades=2)
        # Only TRADE-001 and TRADE-002 should appear.
        assert "TRADE-001" in digest
        assert "TRADE-002" in digest
        assert "TRADE-003" not in digest


# ----- Phase 6 — summary_stats / by-hour / by-regime ------------------------


def test_summary_stats_zero_trades_is_empty_not_raise(tmp_path: Path) -> None:
    """Fresh deploy has no round-trips; CLI must not crash."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # ensure schema exists even without any rows
        RoundTripReconciler(journal)
        s = JournalViews(journal).summary_stats()
        assert isinstance(s, SummaryStats)
        assert s.n_trades == 0
        assert s.win_rate == 0.0
        assert s.sharpe_annualized == 0.0
        assert s.max_drawdown_usd == 0.0


def test_summary_stats_counts_wins_losses_and_pnl(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="w1", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=10.0, pnl_pct=0.01,
              exit_ts=base)
        _seed(journal, trade_id="w2", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=20.0, pnl_pct=0.02,
              exit_ts=base + timedelta(hours=1))
        _seed(journal, trade_id="l1", symbol="MSFT", pattern_hits=["y"],
              bucket=PnlBucket.LOSS, pnl_usd=-15.0, pnl_pct=-0.015,
              exit_ts=base + timedelta(hours=2))
        s = JournalViews(journal).summary_stats()
        assert s.n_trades == 3
        assert s.n_wins == 2
        assert s.n_losses == 1
        assert s.total_pnl_usd == 15.0
        assert s.avg_win_usd == 15.0  # (10 + 20) / 2
        assert s.avg_loss_usd == -15.0
        assert s.win_rate == 2 / 3
        assert s.expectancy_usd == 5.0  # 15 / 3


def test_summary_stats_max_drawdown_finds_largest_pullback(tmp_path: Path) -> None:
    """Cumulative P&L: +10, +30 (peak), +15 (drawdown 15), -5 (drawdown 35)."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i, pnl in enumerate([10.0, 20.0, -15.0, -20.0]):
            bucket = PnlBucket.WIN if pnl > 0 else PnlBucket.LOSS
            _seed(journal, trade_id=f"t-{i}", symbol="AAPL", pattern_hits=["x"],
                  bucket=bucket, pnl_usd=pnl, pnl_pct=pnl / 1000.0,
                  exit_ts=base + timedelta(hours=i))
        s = JournalViews(journal).summary_stats()
        # Peak after trade 2 = +30; lowest after trade 4 = -5; max DD = 35.
        assert s.max_drawdown_usd == 35.0


def test_summary_stats_sharpe_zero_with_single_day(tmp_path: Path) -> None:
    """Sharpe needs ≥2 distinct days of returns to be meaningful — return
    0.0 rather than dividing by zero on a single-day journal."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i, pnl in enumerate([10.0, -5.0]):
            bucket = PnlBucket.WIN if pnl > 0 else PnlBucket.LOSS
            _seed(journal, trade_id=f"t-{i}", symbol="AAPL", pattern_hits=["x"],
                  bucket=bucket, pnl_usd=pnl, pnl_pct=pnl / 1000.0,
                  exit_ts=base + timedelta(hours=i))
        assert JournalViews(journal).summary_stats().sharpe_annualized == 0.0


def test_win_rate_by_hour_groups_by_entry_hour(tmp_path: Path) -> None:
    """Two trades opened at 14:00 UTC, one at 18:00 UTC, mixed outcomes."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # entry_ts is exit_ts - 1h in _seed, so to get entry hour H we pass
        # exit_ts at H+1.
        _seed(journal, trade_id="a", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
              exit_ts=datetime(2026, 4, 1, 15, 0, tzinfo=UTC))
        _seed(journal, trade_id="b", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.LOSS, pnl_usd=-1.0, pnl_pct=-0.01,
              exit_ts=datetime(2026, 4, 1, 15, 30, tzinfo=UTC))
        _seed(journal, trade_id="c", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=2.0, pnl_pct=0.02,
              exit_ts=datetime(2026, 4, 1, 19, 0, tzinfo=UTC))
        rows = JournalViews(journal).win_rate_by_hour()
        as_dict = {hour: (n, wins, wr) for hour, n, wins, wr in rows}
        assert as_dict[14] == (2, 1, 0.5)
        assert as_dict[18] == (1, 1, 1.0)


def test_win_rate_by_regime_buckets_unknown(tmp_path: Path) -> None:
    """Trades without a regime tag end up under 'unknown'."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed(journal, trade_id="a", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=1.0, pnl_pct=0.01,
              exit_ts=base, regime="risk_on_low_vol")
        _seed(journal, trade_id="b", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.LOSS, pnl_usd=-1.0, pnl_pct=-0.01,
              exit_ts=base + timedelta(hours=1), regime="risk_on_low_vol")
        _seed(journal, trade_id="c", symbol="AAPL", pattern_hits=["x"],
              bucket=PnlBucket.WIN, pnl_usd=2.0, pnl_pct=0.02,
              exit_ts=base + timedelta(hours=2))  # no regime
        rows = JournalViews(journal).win_rate_by_regime()
        as_dict = {regime: (n, wins, wr) for regime, n, wins, wr in rows}
        assert as_dict["risk_on_low_vol"] == (2, 1, 0.5)
        assert as_dict["unknown"] == (1, 1, 1.0)
