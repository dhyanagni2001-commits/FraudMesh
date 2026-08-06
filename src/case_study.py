"""
Phase 5: Pull out concrete flagged fraud-ring communities and summarize them
in human-readable form.

This is the most interview-defensible artifact in the whole project: a
metric like "PR-AUC 0.82" is abstract, but "these 14 transactions across 6
hours all shared one device fingerprint and averaged $340, twelve of them
labeled fraud" is a concrete, explainable finding that anyone can evaluate
without trusting a black-box score.
"""
from __future__ import annotations

import json
import os
import sys

import networkx as nx
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import engineer_features, load_raw, make_synthetic_dataset
from src.graph_builder import build_entity_graph


def summarize_component(df: pd.DataFrame, component: set, G: nx.Graph) -> dict:
    idx = list(component)
    sub = df.loc[idx]

    entity_types = set()
    for u, v, data in G.subgraph(idx).edges(data=True):
        entity_types.add(data.get("entity_type", "unknown"))

    return {
        "n_transactions": len(idx),
        "n_fraud": int(sub[config.TARGET_COL].sum()) if config.TARGET_COL in sub else None,
        "fraud_rate": float(sub[config.TARGET_COL].mean()) if config.TARGET_COL in sub else None,
        "avg_amount": float(sub["TransactionAmt"].mean()),
        "total_amount": float(sub["TransactionAmt"].sum()),
        "time_span_hours": float(
            (sub[config.TIME_COL].max() - sub[config.TIME_COL].min()) / 3600
        ),
        "shared_entity_types": sorted(entity_types),
        "unique_cards": int(sub["card1"].nunique()) if "card1" in sub else None,
        "unique_devices": int(sub["DeviceInfo"].nunique()) if "DeviceInfo" in sub else None,
    }


def find_top_rings(df: pd.DataFrame, G: nx.Graph, top_n: int = 5,
                    min_size: int = 5) -> list[dict]:
    """Return the top_n largest connected components with size >= min_size,
    ranked by fraud rate then size — these are the candidate fraud rings."""
    components = [c for c in nx.connected_components(G) if len(c) >= min_size]
    summaries = [summarize_component(df, c, G) for c in components]
    summaries.sort(key=lambda s: (s.get("fraud_rate") or 0, s["n_transactions"]),
                    reverse=True)
    return summaries[:top_n]


def print_ring_report(summaries: list[dict]):
    print(f"Found {len(summaries)} candidate fraud-ring communities:\n")
    for i, s in enumerate(summaries, 1):
        print(f"Ring #{i}")
        print(f"  transactions:        {s['n_transactions']}")
        print(f"  fraud rate:          {s['fraud_rate']:.1%}" if s['fraud_rate'] is not None else "  fraud rate: n/a")
        print(f"  avg amount:          ${s['avg_amount']:.2f}")
        print(f"  total amount:        ${s['total_amount']:.2f}")
        print(f"  time span:           {s['time_span_hours']:.1f} hours")
        print(f"  unique cards:        {s['unique_cards']}")
        print(f"  unique devices:      {s['unique_devices']}")
        print(f"  linked via:          {', '.join(s['shared_entity_types'])}")
        print()


def run(use_synthetic: bool = False, sample_frac: float | None = None,
        out_path: str = os.path.join(config.RESULTS_DIR, "case_study_rings.json")):
    if use_synthetic:
        df = make_synthetic_dataset(n_rows=20000)
    else:
        df = load_raw(sample_frac=sample_frac)

    df = engineer_features(df)
    G = build_entity_graph(df)

    summaries = find_top_rings(df, G, top_n=5, min_size=5)
    print_ring_report(summaries)

    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved ring report to {out_path}")
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()
    run(use_synthetic=args.synthetic, sample_frac=args.sample_frac)
