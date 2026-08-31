import networkx as nx
import numpy as np
import pandas as pd

from src.graph_builder import (
    add_graph_stat_features,
    build_entity_graph,
    graph_summary,
    to_pyg_edge_index,
)


def _toy_df():
    # 5 rows sharing card1="A" (a "ring"), 3 rows with unique card1 values,
    # plus one NaN link value that must not produce any edge.
    return pd.DataFrame({
        "card1": ["A", "A", "A", "A", "A", "B", "C", "D", np.nan],
        "DeviceInfo": [f"dev{i}" for i in range(9)],
        "isFraud": [1, 1, 1, 1, 0, 0, 0, 0, 0],
    })


def test_build_entity_graph_links_shared_entities():
    df = _toy_df()
    G = build_entity_graph(df, link_columns=["card1"])
    # The 5 rows sharing card1="A" should land in one connected component.
    components = list(nx.connected_components(G))
    ring_component = next(c for c in components if 0 in c)
    assert ring_component == {0, 1, 2, 3, 4}


def test_build_entity_graph_respects_max_edges_per_entity():
    df = _toy_df()
    # Cap below the ring size (5) — the "A" ring should get NO edges at all.
    G = build_entity_graph(df, link_columns=["card1"], max_edges_per_entity=3)
    assert G.number_of_edges() == 0


def test_build_entity_graph_skips_nan_link_values():
    df = _toy_df()
    G = build_entity_graph(df, link_columns=["card1"])
    # Row 8 has card1=NaN and no other shared entity — must stay isolated.
    assert G.degree(8) == 0


def test_graph_summary_fraud_rate_climbs_with_component_size():
    """Core validation claim from the README: fraud rate should be higher
    in larger connected components than in isolated (size-1) ones, on
    synthetic data with injected fraud rings."""
    from src.data_prep import engineer_features, make_synthetic_dataset

    df = engineer_features(make_synthetic_dataset(n_rows=5000, seed=0))
    G = build_entity_graph(df)
    summary = graph_summary(G, df, size_thresholds=(1, 10))
    rate_all = summary["fraud_rate_by_min_component_size"][1]
    rate_large = summary["fraud_rate_by_min_component_size"][10]
    assert rate_large > rate_all


def test_add_graph_stat_features_columns_present(synthetic_df):
    G = build_entity_graph(synthetic_df)
    out = add_graph_stat_features(synthetic_df, G)
    for col in ("graph_degree", "graph_component_size", "graph_pagerank"):
        assert col in out.columns
    assert (out["graph_component_size"] >= 1).all()


def test_to_pyg_edge_index_edgeless_graph_falls_back_to_self_loops():
    G = nx.Graph()
    G.add_nodes_from(range(4))
    edge_index = to_pyg_edge_index(G, num_nodes=4)
    assert edge_index.shape == (2, 4)
    assert (edge_index[0] == edge_index[1]).all()
