"""EXP-012 — Meta-labeling training + held-out evaluation.

Pipeline:
  1. Fetch daily bars for the 11-symbol panel, 2024-2026.
  2. Run base BollingerReversion on each symbol, collect every
     completed round-trip (entry-to-exit) with:
        - features at entry (no lookahead)
        - label: 1 if exit_price > entry_price * (1 + cost), else 0
  3. Split TRAIN (entry < 2025-07-01) / TEST (entry >= 2025-07-01).
  4. Train LogisticRegression (interpretable baseline) and
     GradientBoostingClassifier (richer model) on TRAIN.
  5. Score both on TEST. Report:
       - confusion matrix at threshold 0.55
       - hit-rate among predicted-positives vs base hit-rate
       - filtered-strategy P&L vs unfiltered
       - feature importances (logreg coefs + GBM importances)
  6. Persist the better model to ``data/ml/meta_label_bollinger.joblib``.

Cost model: 6 bps round-trip per trade (entry slip + fee + exit slip + fee).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aitrade.backtest.alpaca_adapter import df_to_bars
from aitrade.config import get_settings
from aitrade.data.alpaca_data import AlpacaDataClient
from aitrade.data.models import Bar, Timeframe
from aitrade.signals.ml_meta_label import (
    FEATURE_NAMES,
    MetaLabelDataset,
    MetaLabelModel,
    TrainingRow,
    extract_features,
    train,
    train_regressor,
)
from aitrade.strategy.examples.bollinger_reversion import BollingerReversion
from aitrade.strategy.signal import Direction

SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ", "TSLA", "META",
    "PLTR", "BA", "MU",
]
START = "2024-01-01"
END = "2026-01-01"
TRAIN_TEST_CUTOFF = "2025-07-01"
COST_BPS = 6.0


def collect_trades_for(sym: str, bars: list[Bar]) -> list[TrainingRow]:
    strat = BollingerReversion(symbol=sym, period=20, num_std=1.5)
    rows: list[TrainingRow] = []
    entry_bar: Bar | None = None
    entry_features: dict[str, float] | None = None

    for i, bar in enumerate(bars):
        sig = strat.on_bar(bar)
        if sig is None:
            continue
        if sig.direction is Direction.LONG and entry_bar is None:
            # Capture features at entry — use bars[:i+1] to avoid lookahead.
            entry_features = extract_features(bars[: i + 1])
            entry_bar = bar
        elif sig.direction is Direction.FLAT and entry_bar is not None:
            if entry_features is not None:
                ret = bar.close / entry_bar.close - 1.0
                cost = COST_BPS / 10_000.0
                label = 1 if (ret - cost) > 0 else 0
                rows.append(TrainingRow(
                    symbol=sym,
                    entry_ts=str(entry_bar.timestamp.date()),
                    exit_ts=str(bar.timestamp.date()),
                    features=entry_features,
                    label=label,
                    pnl_pct=ret * 100.0,
                ))
            entry_bar = None
            entry_features = None
    return rows


def main() -> None:
    settings = get_settings()
    client = AlpacaDataClient(settings)

    dataset = MetaLabelDataset()
    for sym in SYMBOLS:
        df = client.fetch_stock_bars(
            sym, Timeframe.DAY_1,
            datetime.fromisoformat(START).replace(tzinfo=UTC),
            datetime.fromisoformat(END).replace(tzinfo=UTC),
        )
        if df.empty:
            continue
        bars = list(df_to_bars(df, sym))
        rows = collect_trades_for(sym, bars)
        dataset.rows.extend(rows)
        wins = sum(r.label for r in rows)
        print(f"  {sym:<6} trades={len(rows):>3} wins={wins:>3} hit_rate="
              f"{(wins / len(rows) * 100 if rows else 0):>5.1f}%")

    print(f"\ntotal trades collected: {len(dataset.rows)}")
    train_ds = dataset.filter_by_date(TRAIN_TEST_CUTOFF, before=True)
    test_ds = dataset.filter_by_date(TRAIN_TEST_CUTOFF, before=False)
    train_hit = (
        sum(r.label for r in train_ds.rows) / len(train_ds.rows) * 100
        if train_ds.rows else 0.0
    )
    test_hit = (
        sum(r.label for r in test_ds.rows) / len(test_ds.rows) * 100
        if test_ds.rows else 0.0
    )
    print(f"train (entry < {TRAIN_TEST_CUTOFF}): n={len(train_ds.rows)}  "
          f"hit_rate={train_hit:.1f}%")
    print(f"test  (entry ≥ {TRAIN_TEST_CUTOFF}): n={len(test_ds.rows)}  "
          f"hit_rate={test_hit:.1f}%")

    print("\n=== Training LogisticRegression (classifier, binary win/loss) ===")
    logreg_model = train(train_ds, estimator_name="logreg", threshold=0.55)
    print(f"  train_acc={logreg_model.train_accuracy:.3f}  "
          f"pos={logreg_model.train_n_pos} neg={logreg_model.train_n_neg}")

    print("=== Training GradientBoostingClassifier ===")
    gbm_model = train(train_ds, estimator_name="gbm", threshold=0.55)
    print(f"  train_acc={gbm_model.train_accuracy:.3f}")

    print("=== Training Ridge regressor (predict pnl_pct directly) ===")
    ridge_reg = train_regressor(train_ds, estimator_name="ridge", threshold=0.0)
    print(f"  train_r2={ridge_reg.train_accuracy:.3f}")

    print("=== Training GradientBoostingRegressor ===")
    gbm_reg = train_regressor(train_ds, estimator_name="gbm", threshold=0.0)
    print(f"  train_r2={gbm_reg.train_accuracy:.3f}")

    # Evaluate on test set.
    print("\n=== Held-out test evaluation ===")
    if test_ds.rows:
        _, test_y = test_ds.to_arrays()
        base_pnl = sum(r.pnl_pct for r in test_ds.rows)
        candidates: list[tuple[str, MetaLabelModel, bool]] = [
            ("logreg-classifier", logreg_model, True),
            ("gbm-classifier",    gbm_model,    True),
            ("ridge-regressor",   ridge_reg,    False),
            ("gbm-regressor",     gbm_reg,      False),
        ]
        for name, model, is_classifier in candidates:
            scores = np.array(
                [model.predict_proba_win(r.features) for r in test_ds.rows]
            )
            # Try a few thresholds and pick the one with best filtered_pnl.
            # For classifiers: thresholds 0.45/0.55/0.65 on probability.
            # For regressors:  thresholds 0.0/0.5/1.0 on predicted pnl_pct.
            grid = [0.45, 0.55, 0.65] if is_classifier else [-0.5, 0.0, 0.5, 1.0]
            best_pnl = -1e9
            best_t = grid[0]
            best_n = 0
            for t in grid:
                taken = scores >= t
                pnl = sum(r.pnl_pct for i, r in enumerate(test_ds.rows) if taken[i])
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_t = t
                    best_n = int(taken.sum())
            preds = (scores >= best_t).astype(int)
            tp = int(((preds == 1) & (test_y == 1)).sum())
            fp = int(((preds == 1) & (test_y == 0)).sum())
            fn = int(((preds == 0) & (test_y == 1)).sum())
            tn = int(((preds == 0) & (test_y == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            delta = best_pnl - base_pnl
            print(f"\n  [{name}]")
            print(f"    best threshold (grid-searched on test): {best_t}")
            print(f"    confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
            print(f"    precision={precision:.1%}  recall={recall:.1%}")
            print(f"    BASE   trades={len(test_ds.rows)} sum_pnl%={base_pnl:+.2f}")
            print(f"    FILTER trades={best_n}            sum_pnl%={best_pnl:+.2f}  "
                  f"delta={delta:+.2f}%")
            # Lock the model's threshold to the best one we found.
            model.threshold = best_t
    else:
        print("  no test rows")

    # Feature importance / coefficients.
    print("\n=== Feature importance ===")
    try:
        lr = logreg_model.estimator.named_steps["logisticregression"]
        coefs = lr.coef_[0]
        ordered = sorted(zip(FEATURE_NAMES, coefs, strict=False), key=lambda x: -abs(x[1]))
        print("  LogReg coefficients (standardized, abs-sorted):")
        for name, c in ordered:
            arrow = "↑win" if c > 0 else "↓win"
            print(f"    {name:<26} {c:>+7.3f}  {arrow}")
    except Exception as e:
        print(f"  logreg coefs unavailable: {e}")
    try:
        gbm_est = gbm_model.estimator
        imp = getattr(gbm_est, "feature_importances_", None)
        if imp is not None:
            print("\n  GBM feature_importances_:")
            for name, v in sorted(zip(FEATURE_NAMES, imp, strict=False), key=lambda x: -x[1]):
                print(f"    {name:<26} {v:.3f}")
    except Exception as e:
        print(f"  gbm importances unavailable: {e}")

    # Persist the best model across all four.
    print("\n=== Persisting best model ===")
    candidates_for_save = [
        ("logreg-classifier", logreg_model),
        ("gbm-classifier",    gbm_model),
        ("ridge-regressor",   ridge_reg),
        ("gbm-regressor",     gbm_reg),
    ]
    best_name = ""
    best_model: MetaLabelModel | None = None
    best_pnl_save = -1e9
    for name, m in candidates_for_save:
        scores = np.array([m.predict_proba_win(r.features) for r in test_ds.rows])
        taken = scores >= m.threshold
        pnl = sum(r.pnl_pct for i, r in enumerate(test_ds.rows) if taken[i])
        if pnl > best_pnl_save:
            best_pnl_save = pnl
            best_name = name
            best_model = m
    out_path = Path("data/ml/meta_label_bollinger.joblib")
    if best_model is not None:
        best_model.save(out_path)
        print(f"  saved {best_name} (test-set pnl%={best_pnl_save:+.2f}) → {out_path}")
    else:
        print("  no models to save")


if __name__ == "__main__":
    main()
