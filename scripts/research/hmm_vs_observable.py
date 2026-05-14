"""EXP-013 — Compare Gaussian HMM regime to observable Markov regime.

Same 9-symbol panel as EXP-009 (which used the observable-state chain).
Fit a 3-state Gaussian HMM per symbol, decode the Viterbi state path,
heuristically name the states by their emission moments, and report:

  - State distribution (HMM vs observable Markov)
  - Whether HMM-labels for the current bar agree with observable-labels
  - Posterior at last bar (HMM gives soft assignment)
  - Per-state mean log-return + variance (helps interpret what HMM learned)

The hypothesis: HMM produces smoother / different state attributions
than the hard-threshold observable version on volatile symbols
(MU/TSLA), but lines up with it on stable ones (SPY/QQQ).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Bar, Timeframe
from aitrade.signals.hmm_regime import fit_hmm
from aitrade.signals.markov_regime import MarkovRegimeChain, Regime

SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ", "TSLA", "META", "IWM"]
NAMES = {
    Regime.TRENDING_UP: "trend_up",
    Regime.MEAN_REVERTING: "mean_rev",
    Regime.STRESSED: "stressed",
}


def observable_state_distribution(bars: list[Bar]) -> dict[str, float]:
    chain = MarkovRegimeChain()
    for b in bars:
        chain.observe(b)
    counts = Counter(chain._states)
    n = len(chain._states) or 1
    return {
        "trending_up": counts[Regime.TRENDING_UP] / n * 100,
        "mean_reverting": counts[Regime.MEAN_REVERTING] / n * 100,
        "stressed": counts[Regime.STRESSED] / n * 100,
        "current": NAMES[chain._states[-1]] if chain._states else "?",
    }


def hmm_state_distribution(bars: list[Bar]) -> dict[str, float] | None:
    snap = fit_hmm(bars, n_states=3)
    if snap is None:
        return None
    names = snap.annotated_state_labels()
    counts = Counter(snap.state_path_full)
    n = len(snap.state_path_full) or 1
    dist: dict[str, float] = {
        "trending_up": 0.0,
        "mean_reverting": 0.0,
        "stressed": 0.0,
        "current": names[snap.state_path_last],
    }
    for state_idx, c in counts.items():
        name = names[state_idx]
        if name in dist:
            dist[name] = c / n * 100
    # Per-state moments (mean return, variance of return).
    dist["state_means"] = [round(m[0] * 100, 3) for m in snap.state_means]
    dist["state_vars"] = [round(v[0] * 10000, 3) for v in snap.state_variances]
    dist["last_posterior"] = [round(p, 3) for p in snap.posterior_last]
    return dist


client = AlpacaDataClient(get_settings())

print(f"{'sym':<5}  "
      f"{'OBS_tr':>7} {'OBS_mr':>7} {'OBS_st':>7} {'OBS_cur':>8}  "
      f"{'HMM_tr':>7} {'HMM_mr':>7} {'HMM_st':>7} {'HMM_cur':>10}  "
      f"{'agree?':>7}")
for sym in SYMBOLS:
    df = client.fetch_stock_bars(
        sym, Timeframe.DAY_1,
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    if df.empty:
        continue
    bars = list(df_to_bars(df, sym))
    obs = observable_state_distribution(bars)
    h = hmm_state_distribution(bars)
    if h is None:
        print(f"{sym:<5}  HMM fit failed")
        continue
    agree = "YES" if obs["current"].split("_")[0] == h["current"].split("_")[0] else "no"
    print(f"{sym:<5}  "
          f"{obs['trending_up']:>6.1f}% {obs['mean_reverting']:>6.1f}% "
          f"{obs['stressed']:>6.1f}% {obs['current']:>8}  "
          f"{h['trending_up']:>6.1f}% {h['mean_reverting']:>6.1f}% "
          f"{h['stressed']:>6.1f}% {h['current']:>10}  {agree:>7}")
    print(f"       HMM emission means (×100): {h['state_means']}  "
          f"vars (×10000): {h['state_vars']}  "
          f"posterior at last bar: {h['last_posterior']}")
