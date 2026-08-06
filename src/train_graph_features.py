"""
Phase 3: XGBoost + cheap graph-derived features (degree, component size,
PageRank), computed from the shared-entity graph built in graph_builder.py.

This is the critical ablation checkpoint of the whole project: it measures
how much of GraphSAGE's eventual lift (if any) is captured by cheap,
non-learned graph statistics, versus requiring the full message-passing
model. Both numbers should be reported side by side in the README ablation
table — a model that only slightly beats this is hard to justify in
production given GraphSAGE's added serving complexity.

IMPORTANT (leakage note): the graph here is built over the full dataset
(train + test) using structural entity relationships (which card/device a
transaction used), not the fraud label. Structural graph membership is
known at transaction time in production (you know a transaction's card and
device before you know if it's fraud), so this does not leak future labels.
What WOULD leak is computing per-component fraud rate from data that
includes the test period — we deliberately avoid that feature here and only
use label-free graph statistics (degree, component size, PageRank).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import (engineer_features, get_feature_columns,
                            load_raw, make_synthetic_dataset, time_aware_split)
from src.graph_builder import add_graph_stat_features, build_entity_graph
from src.metrics import evaluate, format_metrics
from src.train_baseline import train_xgb_baseline


def run(use_synthetic: bool = False, sample_frac: float | None = None,
        out_path: str = os.path.join(config.RESULTS_DIR, "graph_features_metrics.json")):
    if use_synthetic:
        df = make_synthetic_dataset(n_rows=20000)
    else:
        df = load_raw(sample_frac=sample_frac)

    df = engineer_features(df)

    # Build the graph over the FULL dataset (structural entity links only,
    # label-free) before splitting — this mirrors production, where the
    # entity graph is a standing structure updated continuously, not
    # something re-derived per train/test boundary.
    G = build_entity_graph(df)
    df = add_graph_stat_features(df, G)

    train, test = time_aware_split(df)
    # get_feature_columns already picks up graph_degree/graph_component_size/
    # graph_pagerank since they're numeric columns added to df above.
    feature_cols = get_feature_columns(df)

    model, scores = train_xgb_baseline(train, test, feature_cols)
    metrics = evaluate(test[config.TARGET_COL], scores, config.FPR_TARGETS)
    print(format_metrics("xgb_graph_features", metrics))

    top_features = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )[:10]
    print("\nTop 10 features:")
    for name, importance in top_features:
        print(f"  {name}: {importance:.4f}")

    result = {"model": "xgb_graph_features", "metrics": metrics,
              "n_train": len(train), "n_test": len(test),
              "top_features": [(n, float(i)) for n, i in top_features]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics to {out_path}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()
    run(use_synthetic=args.synthetic, sample_frac=args.sample_frac)
