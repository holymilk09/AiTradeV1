"""Meta-labeling ML filter for bollinger entries — Lopez de Prado pattern.

Premise. EXP-011 showed the LLM-as-filter pattern fails because the LLM
imports common-sense risk priors that are *anti-correlated* with where
bollinger has its edge. A model trained on bollinger's own past wins
and losses won't have that problem — it learns the actual edge
structure rather than borrowed wisdom.

This is the "meta-labeling" technique from M. Lopez de Prado's
*Advances in Financial Machine Learning* (2018). The primary model
(bollinger) decides *when* to consider a trade. The secondary model
(this one) decides whether the primary model's signal is likely to
*win* given the current setup. The two-stage architecture lets each
stage do what it's best at.

Feature extraction is pure (no I/O, no lookahead). The same function
is used at training time (over historical bars) and at inference time
(in a live strategy wrapping bollinger). Trained models persist as
joblib artifacts in ``data/ml/`` and are out of git scope (the
training script regenerates them from current bar data, which is the
right life-cycle for an adaptive filter).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aitrade.data.models import Bar
from aitrade.signals.markov_regime import (
    MIN_BARS_FOR_CLASSIFICATION,
    Regime,
    classify_state,
)
from aitrade.strategy.indicators import (
    rolling_vwap,
    rsi,
    volume_ratio,
)

# Feature names in fixed order. Used by both training and inference so the
# matrix columns match. NEVER reorder — only append.
FEATURE_NAMES: tuple[str, ...] = (
    "rsi_14",
    "ret_5d",
    "ret_20d",
    "rvol_20d_ann",
    "dist_from_60d_high",
    "vol_ratio_20d",
    "vwap_distance_pct",
    "z_score_bollinger",
    "markov_trending_up",
    "markov_mean_reverting",
    "markov_stressed",
)

# Minimum bars needed to compute every feature. Bigger of 60 (for the
# distance-from-60d-high) and the markov classification window + 1.
MIN_BARS_FOR_FEATURES = max(60, MIN_BARS_FOR_CLASSIFICATION + 1)


def _safe_log_return(b1: float, b0: float) -> float:
    if b0 <= 0 or b1 <= 0:
        return 0.0
    return math.log(b1 / b0)


def extract_features(bars: Sequence[Bar]) -> dict[str, float] | None:
    """Extract the feature vector at the LAST bar in ``bars``.

    Pure: never reads anything beyond what's in ``bars``. The caller
    must pass only the history up to AND INCLUDING the entry bar; never
    pass future bars or the result will be polluted by lookahead.

    Returns None if there isn't enough history for a complete vector.
    """
    if len(bars) < MIN_BARS_FOR_FEATURES:
        return None

    closes = [b.close for b in bars]
    last_close = closes[-1]

    # Returns.
    ret_5d = last_close / closes[-6] - 1.0
    ret_20d = last_close / closes[-21] - 1.0

    # Realized vol (annualized).
    rets = [
        _safe_log_return(closes[i], closes[i - 1])
        for i in range(max(1, len(closes) - 20), len(closes))
    ]
    if len(rets) >= 2:
        mean_r = sum(rets) / len(rets)
        var_r = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
        rvol = math.sqrt(var_r) * math.sqrt(252.0)
    else:
        rvol = 0.0

    # Distance from 60-day high.
    high_60 = max(closes[-60:])
    dist_high = (last_close / high_60 - 1.0) * 100.0 if high_60 > 0 else 0.0

    # RSI / volume / VWAP using the existing indicator helpers.
    rsi_val = rsi(closes, 14) or 50.0
    vol_r = volume_ratio(list(bars), 20) or 1.0
    vwap = rolling_vwap(list(bars), 20)
    vwap_dist = (last_close / vwap - 1.0) * 100.0 if vwap else 0.0

    # Bollinger z-score (close vs 20-bar mid in stdev units).
    window = closes[-20:]
    mid = sum(window) / 20
    var = sum((c - mid) ** 2 for c in window) / 20
    sd = math.sqrt(var)
    z = (last_close - mid) / sd if sd > 0 else 0.0

    # Markov regime (one-hot).
    return_20d_for_state = ret_20d
    state = classify_state(return_20d_for_state, rvol)
    one_hot_trending = 1.0 if state is Regime.TRENDING_UP else 0.0
    one_hot_mean_rev = 1.0 if state is Regime.MEAN_REVERTING else 0.0
    one_hot_stressed = 1.0 if state is Regime.STRESSED else 0.0

    return {
        "rsi_14": rsi_val,
        "ret_5d": ret_5d * 100.0,  # store as %
        "ret_20d": ret_20d * 100.0,
        "rvol_20d_ann": rvol * 100.0,
        "dist_from_60d_high": dist_high,
        "vol_ratio_20d": vol_r,
        "vwap_distance_pct": vwap_dist,
        "z_score_bollinger": z,
        "markov_trending_up": one_hot_trending,
        "markov_mean_reverting": one_hot_mean_rev,
        "markov_stressed": one_hot_stressed,
    }


def features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a 1xD row in the fixed column order."""
    row = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
    return row.reshape(1, -1)


@dataclass
class TrainingRow:
    symbol: str
    entry_ts: str   # ISO date
    exit_ts: str
    features: dict[str, float]
    label: int      # 1 if winning trade (net of cost), 0 otherwise
    pnl_pct: float  # the realized return for the trade


@dataclass
class MetaLabelDataset:
    rows: list[TrainingRow] = field(default_factory=list)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.rows:
            return np.empty((0, len(FEATURE_NAMES))), np.empty((0,))
        x = np.vstack([features_to_array(r.features) for r in self.rows])
        y = np.array([r.label for r in self.rows], dtype=int)
        return x, y

    def filter_by_date(self, cutoff_iso: str, before: bool) -> MetaLabelDataset:
        if before:
            keep = [r for r in self.rows if r.entry_ts < cutoff_iso]
        else:
            keep = [r for r in self.rows if r.entry_ts >= cutoff_iso]
        return MetaLabelDataset(rows=keep)


@dataclass
class MetaLabelModel:
    """Wrapper around a fitted sklearn estimator + the feature schema.

    The estimator must expose ``predict_proba``. We persist via joblib so
    the on-disk artifact is independent of internal sklearn class names.
    """

    estimator: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES
    threshold: float = 0.5
    train_n_pos: int = 0
    train_n_neg: int = 0
    train_accuracy: float = 0.0

    def predict_proba_win(self, features: dict[str, float]) -> float:
        """Returns the model's score for a setup.

        For classifiers this is the P(win) from ``predict_proba``.
        For regressors it's the predicted pnl_pct from ``predict`` — the
        ``threshold`` field is then interpreted as the minimum expected
        return required to take the trade.
        """
        x = features_to_array(features)
        if hasattr(self.estimator, "predict_proba"):
            return float(self.estimator.predict_proba(x)[0, 1])
        return float(self.estimator.predict(x)[0])

    def should_take(self, features: dict[str, float]) -> bool:
        return self.predict_proba_win(features) >= self.threshold

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self.estimator,
                "feature_names": self.feature_names,
                "threshold": self.threshold,
                "train_n_pos": self.train_n_pos,
                "train_n_neg": self.train_n_neg,
                "train_accuracy": self.train_accuracy,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> MetaLabelModel:
        import joblib
        payload = joblib.load(path)
        return cls(
            estimator=payload["estimator"],
            feature_names=tuple(payload["feature_names"]),
            threshold=float(payload["threshold"]),
            train_n_pos=int(payload.get("train_n_pos", 0)),
            train_n_neg=int(payload.get("train_n_neg", 0)),
            train_accuracy=float(payload.get("train_accuracy", 0.0)),
        )


def train(
    dataset: MetaLabelDataset,
    *,
    estimator_name: str = "logreg",
    threshold: float = 0.55,
) -> MetaLabelModel:
    """Fit a meta-label classifier (binary win/loss) on the dataset."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x, y = dataset.to_arrays()
    if x.shape[0] < 10:
        raise ValueError(
            f"too few training rows ({x.shape[0]}); need ≥10 to meaningfully fit"
        )

    if estimator_name == "logreg":
        est = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000, random_state=42,
            ),
        )
    elif estimator_name == "gbm":
        est = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
        )
    else:
        raise ValueError(f"unknown estimator {estimator_name}")

    est.fit(x, y)
    train_accuracy = float(est.score(x, y))
    return MetaLabelModel(
        estimator=est,
        threshold=threshold,
        train_n_pos=int((y == 1).sum()),
        train_n_neg=int((y == 0).sum()),
        train_accuracy=train_accuracy,
    )


def train_regressor(
    dataset: MetaLabelDataset,
    *,
    estimator_name: str = "ridge",
    threshold: float = 0.0,
) -> MetaLabelModel:
    """Fit a meta-label REGRESSOR predicting per-trade pnl_pct.

    Why this matters. Binary classification weights small winners and big
    winners identically, which biases the model toward "shallow setups
    win more often" — the prudent-trader anti-edge documented in EXP-011.
    Regression on pnl_pct lets the model see that ONE deep-discount big
    winner outweighs three shallow scratches, and learn to keep them.

    The resulting MetaLabelModel stores a regressor; ``predict_proba_win``
    returns the predicted pnl_pct (not a probability — interpreted as an
    expected-return score), and ``threshold`` is the minimum expected
    return needed to take the trade.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not dataset.rows:
        raise ValueError("empty dataset")
    x = np.vstack([features_to_array(r.features) for r in dataset.rows])
    y_reg = np.array([r.pnl_pct for r in dataset.rows], dtype=float)
    if x.shape[0] < 10:
        raise ValueError(f"too few training rows ({x.shape[0]})")

    if estimator_name == "ridge":
        est = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
    elif estimator_name == "gbm":
        est = GradientBoostingRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
        )
    else:
        raise ValueError(f"unknown regressor {estimator_name}")

    est.fit(x, y_reg)
    # For a regressor, "accuracy" is meaningless — use R² on training as
    # a sanity signal (overfit-prone for n<200).
    r2 = float(est.score(x, y_reg))
    return MetaLabelModel(
        estimator=est,
        threshold=threshold,
        train_n_pos=int((y_reg > 0).sum()),
        train_n_neg=int((y_reg <= 0).sum()),
        train_accuracy=r2,
    )
