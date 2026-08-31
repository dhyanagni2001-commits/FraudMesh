"""
Build a shared-entity graph over transactions.

Two transactions get an edge if they share a value in one of
config.ENTITY_LINK_COLUMNS (card1, DeviceInfo, P_emaildomain, addr1).
This is the structural bet of the whole project: fraud rings look like
dense, reused clusters of these shared identifiers, which a row-wise
model (XGBoost on raw features) cannot see directly.

We build the graph with networkx for inspection/community detection, and
export edge_index tensors for PyTorch Geometric training.
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_entity_graph(df: pd.DataFrame, link_columns=None,
                        max_edges_per_entity: int = config.MAX_EDGES_PER_ENTITY) -> nx.Graph:
    """Build a graph where nodes are transaction indices (0..len(df)-1) and
    edges connect transactions sharing a value in any link column.

    max_edges_per_entity skips any single entity value linking more rows
    than this, treating it as too generic to be informative (e.g. a
    placeholder/default value rather than a genuine shared identity) —
    see the comment on config.MAX_EDGES_PER_ENTITY for the reasoning and
    why this is NOT primarily a computational safeguard: edges here are
    built hub-and-spoke (below), so cost is O(n) per entity, not the
    O(n^2) a full clique would need.
    """
    link_columns = link_columns or config.ENTITY_LINK_COLUMNS
    link_columns = [c for c in link_columns if c in df.columns]

    G = nx.Graph()
    G.add_nodes_from(df.index)

    for col in link_columns:
        groups = df.groupby(col).groups
        n_skipped_entities = 0
        n_skipped_rows = 0
        for value, idx in groups.items():
            if pd.isna(value):
                continue
            idx = list(idx)
            if len(idx) < 2:
                continue
            if len(idx) > max_edges_per_entity:
                # skip overly generic entities (e.g. a placeholder device string)
                n_skipped_entities += 1
                n_skipped_rows += len(idx)
                continue
            # connect this group as a small clique via a hub-and-spoke
            # star instead of full clique to keep edge count linear, not
            # quadratic, per entity — sufficient for community detection.
            hub = idx[0]
            for other in idx[1:]:
                G.add_edge(hub, other, entity_type=col, entity_value=str(value))

        if n_skipped_entities:
            # Silent on the real dataset this would otherwise hide exactly
            # how much of the graph's potential signal never gets linked —
            # e.g. a handful of very common real card1 values touching tens
            # of thousands of rows, none of which get an edge from this
            # column at all. Worth knowing when interpreting a weak or flat
            # graph_features/GraphSAGE result.
            print(f"  build_entity_graph: skipped {n_skipped_entities} '{col}' "
                  f"value(s) above max_edges_per_entity={max_edges_per_entity} "
                  f"({n_skipped_rows} rows got no '{col}' edges at all)")

    return G


def graph_summary(G: nx.Graph, df: pd.DataFrame,
                   size_thresholds=(1, 2, 5, 10, 20)) -> dict:
    """Sanity-check the graph: do fraud transactions cluster more than
    legitimate ones? This is the validation to run before trusting the graph
    is worth modeling on.

    The key diagnostic is fraud rate as a function of component size — if
    the graph carries real signal, fraud rate should climb sharply in larger
    components (dense clusters of reused cards/devices = fraud rings), well
    above the base rate seen in isolated single-transaction components.
    """
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]

    fraud_col = df[config.TARGET_COL] if config.TARGET_COL in df.columns else None
    overall_fraud_rate = float(fraud_col.mean()) if fraud_col is not None else None

    fraud_rate_by_min_size = {}
    if fraud_col is not None:
        for min_size in size_thresholds:
            rates = [
                fraud_col.loc[list(c)].mean()
                for c in components if len(c) >= min_size
            ]
            fraud_rate_by_min_size[min_size] = (
                float(np.mean(rates)) if rates else None
            )

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": len(components),
        "n_isolated_nodes": sum(1 for c in comp_sizes if c == 1),
        "largest_component_size": max(comp_sizes) if comp_sizes else 0,
        "overall_fraud_rate": overall_fraud_rate,
        "fraud_rate_by_min_component_size": fraud_rate_by_min_size,
    }


def add_graph_stat_features(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Compute cheap per-node graph statistics to use as extra XGBoost
    features (Phase 3: graph-augmented XGBoost baseline). These are the
    features that let us measure how much signal is captured WITHOUT a full
    GNN, before paying for GraphSAGE.

    Deliberately NOT included: a raw connected-component ID. It's tempting
    to add ("which cluster is this transaction in?") but it's a leakage
    trap — the graph is built over train+test combined, so a component ID
    is shared verbatim between train and test rows in the same cluster.
    XGBoost can then split exactly on that ID and memorize the training
    fraud rate of a component, which "predicts" test rows in the same
    component perfectly for reasons that have nothing to do with
    generalization. component_size/degree/pagerank are safe because they're
    real-valued structural statistics, not identity keys a tree can pin to.
    """
    df = df.copy()

    degree = dict(G.degree())
    df["graph_degree"] = df.index.map(degree).fillna(0)

    components = list(nx.connected_components(G))
    comp_size = {}
    for comp in components:
        for node in comp:
            comp_size[node] = len(comp)
    df["graph_component_size"] = df.index.map(comp_size).fillna(1)

    # PageRank as a cheap proxy for "central to a dense cluster"
    if G.number_of_edges() > 0:
        pr = nx.pagerank(G, max_iter=50)
        df["graph_pagerank"] = df.index.map(pr).fillna(0)
    else:
        df["graph_pagerank"] = 0.0

    return df


def to_pyg_edge_index(G: nx.Graph, num_nodes: int):
    """Convert to a PyTorch Geometric edge_index tensor (undirected, both
    directions included, as PyG expects for message passing)."""
    import torch

    edges = list(G.edges())
    if not edges:
        # No edges found — return a self-loop only graph so PyG doesn't choke
        src = np.arange(num_nodes)
        edge_index = np.vstack([src, src])
    else:
        src = np.array([e[0] for e in edges])
        dst = np.array([e[1] for e in edges])
        edge_index = np.vstack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src]),
        ])
    return torch.tensor(edge_index, dtype=torch.long)


if __name__ == "__main__":
    from data_prep import engineer_features, make_synthetic_dataset

    df = make_synthetic_dataset(n_rows=5000)
    df = engineer_features(df)

    G = build_entity_graph(df)
    summary = graph_summary(G, df)
    print("Graph summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    df_with_stats = add_graph_stat_features(df, G)
    print("\nSample graph-derived features:")
    print(df_with_stats[["isFraud", "graph_degree", "graph_component_size",
                          "graph_pagerank"]].head(10))
