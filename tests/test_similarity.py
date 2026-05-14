"""Similarity retrieval — score weighting, ordering, and JSON tolerance."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aitrade.journal.round_trips import (
    ExitReason,
    PnlBucket,
    RoundTripReconciler,
    TradeRoundTrip,
)
from aitrade.journal.similarity import SimilarityQuery, SimilarTradesFinder
from aitrade.logging.trade_logger import TradeLogger


def _seed_round_trip(
    journal: TradeLogger,
    *,
    trade_id: str,
    symbol: str,
    pattern_hits: list[str],
    regime: str | None,
    exit_ts: datetime,
    pattern_hits_raw: str | None = None,
    market_snapshot_raw: str | None = None,
) -> None:
    """Insert a synthetic round-trip row directly into the SQLite mirror."""
    # Ensure the schema exists. Constructing a reconciler is enough.
    RoundTripReconciler(journal)

    if pattern_hits_raw is not None:
        ph_payload: object = pattern_hits_raw
    else:
        ph_payload = json.dumps(pattern_hits)

    if market_snapshot_raw is not None:
        ms_payload: object = market_snapshot_raw
    else:
        ms_payload = json.dumps({"regime": regime}) if regime is not None else None

    entry_ts = exit_ts - timedelta(hours=1)
    journal._conn.execute(  # noqa: SLF001 — fixture seeding
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
            105.0,
            1.0,
            5.0,
            0.05,
            3600,
            PnlBucket.WIN.value,
            ExitReason.UNKNOWN.value,
            None,
            None,
            None,
            ph_payload,
            ms_payload,
        ),
    )


def test_empty_journal_returns_empty(tmp_path: Path) -> None:
    """No rows -> nothing to compare against."""
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # Need to ensure the schema exists.
        RoundTripReconciler(journal)
        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            )
        )
        assert result == []


def test_below_min_matches_returns_empty(tmp_path: Path) -> None:
    """Too few historical trades -> reasoner sees no prior experience."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i in range(2):
            _seed_round_trip(
                journal,
                trade_id=f"t-{i}",
                symbol="AAPL",
                pattern_hits=["breakout"],
                regime="risk_on_low_vol",
                exit_ts=base + timedelta(hours=i),
            )
        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            )
        )
        assert result == []


def test_same_symbol_beats_only_shared_pattern(tmp_path: Path) -> None:
    """A same-symbol match (+3) outscores a row sharing one pattern (+2)."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # Same symbol, no shared patterns, no regime match.
        _seed_round_trip(
            journal,
            trade_id="same-symbol",
            symbol="AAPL",
            pattern_hits=["unrelated"],
            regime="risk_off_high_vol",
            exit_ts=base,
        )
        # Different symbol, one shared pattern, no regime match.
        _seed_round_trip(
            journal,
            trade_id="shared-pattern",
            symbol="MSFT",
            pattern_hits=["breakout"],
            regime="risk_off_high_vol",
            exit_ts=base + timedelta(hours=1),
        )
        # Filler so we clear the min_matches gate.
        _seed_round_trip(
            journal,
            trade_id="filler",
            symbol="GOOG",
            pattern_hits=["filler"],
            regime="risk_off_high_vol",
            exit_ts=base - timedelta(hours=1),
        )

        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            ),
            top_k=2,
        )
        assert [rt.trade_id for rt in result] == ["same-symbol", "shared-pattern"]


def test_pattern_cap_respected(tmp_path: Path) -> None:
    """Sharing 5 patterns shouldn't outscore a same-symbol + same-regime hit."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    five_shared = ["p1", "p2", "p3", "p4", "p5"]
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # Different symbol, 5 shared patterns. Without the cap this would
        # score 2*5 = 10. With cap=2 it scores 2*2 = 4.
        _seed_round_trip(
            journal,
            trade_id="many-patterns",
            symbol="MSFT",
            pattern_hits=five_shared,
            regime="risk_off_high_vol",
            exit_ts=base + timedelta(hours=1),
        )
        # Same symbol + same regime + zero shared patterns: 3 + 1 = 4.
        # Tie on score → recency wins. Make this one strictly higher
        # by also sharing one pattern: 3 + 2 + 1 = 6 > 4.
        _seed_round_trip(
            journal,
            trade_id="same-symbol-regime",
            symbol="AAPL",
            pattern_hits=["p1"],
            regime="risk_on_low_vol",
            exit_ts=base,
        )
        # Filler.
        _seed_round_trip(
            journal,
            trade_id="filler",
            symbol="GOOG",
            pattern_hits=["unrelated"],
            regime="risk_off_high_vol",
            exit_ts=base - timedelta(hours=2),
        )

        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=five_shared,
                market_regime="risk_on_low_vol",
            ),
            top_k=2,
        )
        assert result[0].trade_id == "same-symbol-regime"
        # And that the cap actually fired: change the same-symbol row to
        # a totally unmatching scenario and the many-patterns row should
        # still score only SHARED_PATTERN_W * cap = 4.
        finder_no_anchor = SimilarTradesFinder(journal, min_matches=3)
        score_only_patterns = finder_no_anchor._score(  # noqa: SLF001 — exercising the cap
            cand=finder_no_anchor._load_candidates()[0],  # noqa: SLF001
            query=SimilarityQuery(
                symbol="ZZZZ",
                pattern_hits=five_shared,
                market_regime=None,
            ),
            query_patterns=set(five_shared),
        )
        expected_capped = (
            SimilarTradesFinder.SHARED_PATTERN_W * SimilarTradesFinder.SHARED_PATTERN_CAP
        )
        assert score_only_patterns == expected_capped


def test_ties_broken_by_exit_ts_desc(tmp_path: Path) -> None:
    """Same score → newer exit_ts wins."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        _seed_round_trip(
            journal,
            trade_id="older",
            symbol="AAPL",
            pattern_hits=["breakout"],
            regime="risk_on_low_vol",
            exit_ts=base,
        )
        _seed_round_trip(
            journal,
            trade_id="newer",
            symbol="AAPL",
            pattern_hits=["breakout"],
            regime="risk_on_low_vol",
            exit_ts=base + timedelta(days=1),
        )
        _seed_round_trip(
            journal,
            trade_id="middle",
            symbol="AAPL",
            pattern_hits=["breakout"],
            regime="risk_on_low_vol",
            exit_ts=base + timedelta(hours=12),
        )

        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            ),
            top_k=3,
        )
        assert [rt.trade_id for rt in result] == ["newer", "middle", "older"]


def test_top_k_limits_output(tmp_path: Path) -> None:
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i in range(5):
            _seed_round_trip(
                journal,
                trade_id=f"t-{i}",
                symbol="AAPL",
                pattern_hits=["breakout"],
                regime="risk_on_low_vol",
                exit_ts=base + timedelta(hours=i),
            )
        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            ),
            top_k=2,
        )
        assert len(result) == 2


def test_malformed_json_is_tolerated(tmp_path: Path) -> None:
    """Garbage in a single row's JSON columns shouldn't crash retrieval."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        # One healthy row.
        _seed_round_trip(
            journal,
            trade_id="healthy",
            symbol="AAPL",
            pattern_hits=["breakout"],
            regime="risk_on_low_vol",
            exit_ts=base,
        )
        # Corrupt pattern_hits → row is skipped.
        _seed_round_trip(
            journal,
            trade_id="bad-pattern",
            symbol="AAPL",
            pattern_hits=[],
            regime="risk_on_low_vol",
            exit_ts=base + timedelta(hours=1),
            pattern_hits_raw="{not-json",
        )
        # Corrupt market_snapshot → row still considered, regime treated as None.
        _seed_round_trip(
            journal,
            trade_id="bad-snapshot",
            symbol="AAPL",
            pattern_hits=["breakout"],
            regime="risk_on_low_vol",
            exit_ts=base + timedelta(hours=2),
            market_snapshot_raw="{not-json",
        )
        # Filler so we exceed min_matches=3.
        _seed_round_trip(
            journal,
            trade_id="filler",
            symbol="GOOG",
            pattern_hits=["filler"],
            regime="risk_off_high_vol",
            exit_ts=base - timedelta(hours=1),
        )

        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            ),
            top_k=5,
        )
        ids = [rt.trade_id for rt in result]
        # bad-pattern row was skipped; the other three survived.
        assert "bad-pattern" not in ids
        assert "healthy" in ids
        assert "bad-snapshot" in ids


def test_returns_actual_round_trip_objects(tmp_path: Path) -> None:
    """Round-trips returned must be fully-typed TradeRoundTrip instances."""
    base = datetime(2026, 4, 1, 15, 0, tzinfo=UTC)
    with TradeLogger(log_dir=tmp_path, strategy_id="test") as journal:
        for i in range(3):
            _seed_round_trip(
                journal,
                trade_id=f"t-{i}",
                symbol="AAPL",
                pattern_hits=["breakout"],
                regime="risk_on_low_vol",
                exit_ts=base + timedelta(hours=i),
            )
        finder = SimilarTradesFinder(journal, min_matches=3)
        result = finder.find(
            SimilarityQuery(
                symbol="AAPL",
                pattern_hits=["breakout"],
                market_regime="risk_on_low_vol",
            ),
            top_k=1,
        )
        assert len(result) == 1
        assert isinstance(result[0], TradeRoundTrip)
        assert result[0].symbol == "AAPL"
        assert result[0].pattern_hits == ["breakout"]
