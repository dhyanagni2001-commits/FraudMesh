"""
Phase 1: XGBoost baseline on raw tabular features only (no graph).

This is the floor every later model must beat. Run this first, always, on
any new dataset — a graph-based model that can't beat a well-tuned XGBoost
baseline isn't worth its added complexity, and this project is designed to
prove or disprove that honestly rather than assume it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import xgboost as xgb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import (
    engineer_features,
    get_feature_columns,
    load_raw,
    make_synthetic_dataset,
    time_aware_split,
)
from src.metrics import evaluate, format_metrics


def train_xgb_baseline(train, test, feature_cols, target_col=config.TARGET_COL):
    """Train XGBoost with class-imbalance-aware settings and return the
    fitted model plus test-set predictions."""
    fraud_rate = train[target_col].mean()
    scale_pos_weight = (1 - fraud_rate) / fraud_rate  # standard imbalance correction

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(
        train[feature_cols], train[target_col],
        eval_set=[(test[feature_cols], test[target_col])],
        verbose=False,
    )

    scores = model.predict_proba(test[feature_cols])[:, 1]
    return model, scores


def run(use_synthetic: bool = False, sample_frac: float | None = None,
        out_path: str = os.path.join(config.RESULTS_DIR, "baseline_metrics.json")):
    if use_synthetic:
        df = make_synthetic_dataset(n_rows=20000)
    else:
        df = load_raw(sample_frac=sample_frac)

    df = engineer_features(df)
    train, test = time_aware_split(df)
    feature_cols = get_feature_columns(df)

    model, scores = train_xgb_baseline(train, test, feature_cols)
    metrics = evaluate(test[config.TARGET_COL], scores, config.FPR_TARGETS)
    print(format_metrics("xgb_baseline", metrics))

    top_features = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1],
    )[:10]
    print("\nTop 10 features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

    result = {"model": "xgb_baseline", "metrics": metrics,
              "n_train": len(train), "n_test": len(test),
              "top_features": [(n, float(i)) for n, i in top_features]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
    return model, metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Use generated synthetic data instead of real IEEE-CIS CSVs")
    parser.add_argument("--sample-frac", type=float, default=None,
                         help="Subsample fraction of legit rows for faster local runs")
    args = parser.parse_args(argv)
    run(use_synthetic=args.synthetic, sample_frac=args.sample_frac)


if __name__ == "__main__":
    main()
