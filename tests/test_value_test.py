"""Reasoner value-test harness: A vs B metrics + keep/cut decision."""

from __future__ import annotations

import pandas as pd

from aitrade.signals.value_test import decision_string, value_test


def _setups(returns: list[float], quant_edges: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=len(returns), freq="D"),
            "fwd_return": returns,
            "quant_edge": quant_edges,
        }
    )


def test_value_test_perfect_filter() -> None:
    # Reasoner accepts only winners. Side B should crush side A.
    df = _setups(
        returns=[0.02, -0.01, 0.03, -0.02, 0.04, -0.03],
        quant_edges=[0.9, 0.1, 0.9, 0.1, 0.9, 0.1],
    )
    accept = lambda row: row["quant_edge"] > 0.5  # noqa: E731
    result = value_test(df, accept)

    assert result.side_a.n_trades == 6
    assert result.side_b.n_trades == 3
    assert result.side_b.expectancy > result.side_a.expectancy
    assert result.side_b.win_rate == 1.0
    assert result.acceptance_rate == 0.5


def test_value_test_useless_filter_recommends_cut() -> None:
    # Reasoner accepts everything — same as A, but adds cost.
    df = _setups(
        returns=[0.01, -0.01, 0.02, -0.02],
        quant_edges=[0.9, 0.9, 0.9, 0.9],
    )
    accept = lambda _row: True  # noqa: E731
    result = value_test(df, accept, cost_per_trade=0.005)

    # net_expectancy_b = avg(returns) - cost  -- should not beat A.
    assert result.net_expectancy_b < result.side_a.expectancy
    assert "CUT" in decision_string(result, min_trades=2)


def test_value_test_empty() -> None:
    df = pd.DataFrame({"ts": [], "fwd_return": []})
    result = value_test(df, lambda _r: True)
    assert result.side_a.n_trades == 0
    assert result.side_b.n_trades == 0
    assert result.acceptance_rate == 0.0


def test_decision_insufficient_data() -> None:
    df = _setups([0.01], [0.9])
    result = value_test(df, lambda _r: True)
    assert "INSUFFICIENT_DATA" in decision_string(result, min_trades=50)
