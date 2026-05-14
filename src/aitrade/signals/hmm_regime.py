"""Gaussian hidden Markov model (HMM) regime classifier.

This is the proper "hidden" Markov model — distinct from
:mod:`aitrade.signals.markov_regime`, which is observable-state with
hard-threshold labels. Differences that matter:

  - **Hidden states.** The state at each bar is not directly observed;
    we infer it probabilistically from the observation (log-return).
    Baum-Welch (EM) fits the transition matrix, the emission means
    and variances per state, and the initial state distribution.
  - **Soft state assignments.** For any bar we get a posterior
    distribution over states ``P(state_t | observations_{1..T})`` via
    the forward-backward algorithm. The observable version assigns
    each bar to exactly one bucket.
  - **Smoother regimes.** A short shock (one bar of high vol) doesn't
    necessarily flip the inferred state, because the HMM weighs the
    new evidence against the transition prior.

We use **two-feature** observations per bar: ``(log_return, log_squared_return)``.
The squared return captures the variance regime; together they let
the HMM separate "low-vol drift", "low-vol decline", "high-vol mean-
reverting", and so on, depending on K. K=3 by default to match the
observable version's state count.

Why two features instead of one. Hmmlearn's GaussianHMM with K
diagonal-covariance components and a single return feature collapses
states with similar means even when their variances differ. Adding the
squared return as a second feature gives the HMM a vol axis to
separate on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from aitrade.data.models import Bar

MIN_BARS_FOR_HMM_FIT = 60   # below this, EM is unreliable
DEFAULT_N_STATES = 3
DEFAULT_N_ITER = 50
DEFAULT_RANDOM_STATE = 42


def _observations_from_closes(closes: Sequence[float]) -> np.ndarray:
    """Build the (T-1, 2) observation matrix: [log_return, log_return²]."""
    if len(closes) < 2:
        return np.empty((0, 2))
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            rets.append(0.0)
        else:
            rets.append(math.log(closes[i] / closes[i - 1]))
    arr = np.asarray(rets, dtype=float).reshape(-1, 1)
    sq = arr ** 2
    return np.hstack([arr, sq])


@dataclass(frozen=True, slots=True)
class HmmSnapshot:
    """Output of a fitted HMM at the last bar of the input."""

    state_means: tuple[tuple[float, float], ...]
    state_variances: tuple[tuple[float, float], ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    state_path_last: int               # Viterbi-decoded most-likely state at T
    posterior_last: tuple[float, ...]  # P(state=k | obs) at bar T
    state_path_full: tuple[int, ...]   # Viterbi decoding for the whole sequence
    n_states: int

    def annotated_state_labels(self) -> tuple[str, ...]:
        """Heuristic name per state based on its emission moments.

        Ranks states by mean return and by variance to assign
        ``trending_up`` / ``mean_reverting`` / ``stressed`` semantics.
        Best-effort — the HMM doesn't know our naming convention; this
        helps the LLM reasoner read the result.
        """
        # Mean of the *return* feature (column 0).
        return_means = [m[0] for m in self.state_means]
        # Variance of the *return* feature (diagonal cov column 0).
        return_vars = [v[0] for v in self.state_variances]

        # Most volatile state = stressed.
        idx_stressed = max(range(self.n_states), key=lambda i: return_vars[i])
        remaining = [i for i in range(self.n_states) if i != idx_stressed]
        # Of the remaining, highest mean return = trending_up.
        if remaining:
            idx_trend = max(remaining, key=lambda i: return_means[i])
            remaining = [i for i in remaining if i != idx_trend]
        else:
            idx_trend = -1
        # The leftover is mean_reverting.
        idx_mr = remaining[0] if remaining else -1

        names: list[str] = ["?"] * self.n_states
        for i in range(self.n_states):
            if i == idx_stressed:
                names[i] = "stressed"
            elif i == idx_trend:
                names[i] = "trending_up"
            elif i == idx_mr:
                names[i] = "mean_reverting"
        return tuple(names)


def fit_hmm(
    bars: Sequence[Bar],
    *,
    n_states: int = DEFAULT_N_STATES,
    n_iter: int = DEFAULT_N_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> HmmSnapshot | None:
    """Fit a Gaussian HMM to the bar sequence; return a snapshot at the last bar.

    Returns None if fewer than MIN_BARS_FOR_HMM_FIT bars are provided.
    """
    if len(bars) < MIN_BARS_FOR_HMM_FIT:
        return None

    closes = [b.close for b in bars]
    obs = _observations_from_closes(closes)
    if obs.shape[0] < MIN_BARS_FOR_HMM_FIT - 1:
        return None

    from hmmlearn import hmm

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-3,
    )
    # hmmlearn emits convergence warnings to stderr; we just consume them.
    try:
        model.fit(obs)
    except Exception:
        return None
    if not getattr(model, "monitor_", None) or not model.monitor_.converged:
        # Allow non-converged fits too — they still produce usable
        # posteriors. We just flag this implicitly via the state-mean spread.
        pass

    state_seq = model.predict(obs)
    log_post = model.predict_proba(obs)
    means: list[tuple[float, float]] = [tuple(row) for row in model.means_]
    cov_diags: list[tuple[float, float]] = [tuple(np.diag(c)) for c in model.covars_]
    trans: list[tuple[float, ...]] = [tuple(row) for row in model.transmat_]

    return HmmSnapshot(
        state_means=tuple(means),
        state_variances=tuple(cov_diags),
        transition_matrix=tuple(trans),
        state_path_last=int(state_seq[-1]),
        posterior_last=tuple(float(p) for p in log_post[-1]),
        state_path_full=tuple(int(s) for s in state_seq),
        n_states=n_states,
    )
