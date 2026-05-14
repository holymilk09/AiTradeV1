"""Universe composition: watchlist + movers + optional web, merged by symbol."""

from __future__ import annotations

from datetime import UTC, datetime

from aitrade.discovery.scorer import DiscoveredTicker
from aitrade.discovery.universe import build_universe, merge_sources
from aitrade.discovery.watchlist import (
    DEFAULT_WATCHLIST,
    load_watchlist,
    watchlist_as_discovered,
)


def _t(symbol: str, buzz: float, evidence: list[str] | None = None) -> DiscoveredTicker:
    return DiscoveredTicker(
        symbol=symbol,
        mention_count=1,
        source_weight=1.0,
        recency_minutes=0.0,
        buzz_score=buzz,
        discovered_at=datetime(2026, 4, 24, tzinfo=UTC),
        evidence=evidence or [],
    )


def test_default_watchlist_is_non_empty_and_uppercase() -> None:
    syms = load_watchlist()
    assert syms
    for s in syms:
        assert s == s.upper()
    assert "SPY" in syms
    assert "AAPL" in syms


def test_env_override_replaces_default_watchlist(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AITRADE_WATCHLIST", "abc,DEF, ghi ,abc")
    syms = load_watchlist()
    assert syms == ["ABC", "DEF", "GHI"]


def test_watchlist_as_discovered_assigns_baseline_buzz() -> None:
    out = watchlist_as_discovered(["AAPL", "MSFT"])
    assert {d.symbol for d in out} == {"AAPL", "MSFT"}
    assert all(d.buzz_score == 1.0 for d in out)
    assert all(d.evidence == ["watchlist"] for d in out)


def test_merge_sources_higher_buzz_wins_evidence_concats() -> None:
    a = [_t("AAPL", 1.0, ["watchlist"])]
    b = [_t("AAPL", 0.5, ["mover +3%"])]
    c = [_t("AAPL", 1.5, ["web buzz"])]
    out = merge_sources(a, b, c)
    assert len(out) == 1
    aapl = out[0]
    assert aapl.buzz_score == 1.5  # winner
    assert "watchlist" in aapl.evidence
    assert "mover +3%" in aapl.evidence
    assert "web buzz" in aapl.evidence


def test_merge_sources_distinct_symbols_kept_and_sorted() -> None:
    a = [_t("LOW", 0.3)]
    b = [_t("HIGH", 0.9), _t("MID", 0.5)]
    out = merge_sources(a, b)
    assert [d.symbol for d in out] == ["HIGH", "MID", "LOW"]


def test_build_universe_watchlist_only_no_network() -> None:
    out = build_universe(use_watchlist=True, use_movers=False, use_web=False)
    assert out
    syms = {d.symbol for d in out}
    assert "SPY" in syms or DEFAULT_WATCHLIST[0] in syms


def test_build_universe_with_movers_passes_through() -> None:
    class _StubMovers:
        def find(self) -> list[DiscoveredTicker]:
            return [_t("NVDA", 0.7, ["gainer +5%"])]

    out = build_universe(
        movers=_StubMovers(),  # type: ignore[arg-type]
        use_watchlist=False,
        use_movers=True,
        use_web=False,
    )
    assert any(d.symbol == "NVDA" for d in out)


def test_build_universe_web_off_does_not_call_agent() -> None:
    class _ExplodingAgent:
        def discover(self, top_n: int) -> list[DiscoveredTicker]:
            raise AssertionError("web agent should not be called when use_web=False")

    out = build_universe(
        web_agent=_ExplodingAgent(),  # type: ignore[arg-type]
        use_watchlist=True,
        use_movers=False,
        use_web=False,
    )
    assert out  # watchlist still present
