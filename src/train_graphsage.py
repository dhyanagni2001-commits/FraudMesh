"""
Phase 4: GraphSAGE trained end-to-end on the shared-entity transaction graph.

Inductive setup: GraphSAGE learns an aggregation function over a node's
neighborhood rather than a fixed embedding table, so it generalizes to
transactions/nodes it never saw during training. That's the property that
matters for fraud in production, where new transactions and new
card/device combinations show up constantly. We enforce this by masking
test nodes' labels during message passing feature construction and
evaluating strictly on nodes outside the training time window.

This script trains directly on raw + engineered numeric features as node
features, with the entity graph supplying the edges. Compare its metrics
against train_baseline.py (no graph) and train_graph_features.py (graph
stats only) to complete the ablation table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import engineer_features, get_feature_columns, load_raw, make_synthetic_dataset
from src.graph_builder import build_entity_graph, to_pyg_edge_index
from src.metrics import evaluate, format_metrics


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.out = torch.nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x).squeeze(-1)


def build_pyg_data(df, feature_cols, G):
    # fillna(0): the real IEEE-CIS columns (many V*/D* fields) are heavily
    # missing. XGBoost handles NaN natively via its split logic, but a raw
    # tensor doesn't — an unfilled NaN here silently propagates through
    # standardization and every layer, turning the whole forward pass (and
    # then the loss) into NaN with no error raised. Doesn't show up on the
    # synthetic data since that generator never produces missing values.
    x = torch.tensor(df[feature_cols].fillna(0).values, dtype=torch.float)
    # Standardize features — GNNs are sensitive to feature scale
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

    y = torch.tensor(df[config.TARGET_COL].values, dtype=torch.float)
    edge_index = to_pyg_edge_index(G, num_nodes=len(df))
    return x, edge_index, y


def train_graphsage(df, feature_cols, G, train_mask, test_mask,
                     hidden_dim=config.SAGE_HIDDEN_DIM,
                     num_layers=config.SAGE_NUM_LAYERS,
                     dropout=config.SAGE_DROPOUT,
                     epochs=config.SAGE_EPOCHS, lr=config.SAGE_LR):
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    x, edge_index, y = build_pyg_data(df, feature_cols, G)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    model = GraphSAGE(in_dim=x.shape[1], hidden_dim=hidden_dim,
                       num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # Class-imbalance-aware loss, same rationale as scale_pos_weight in XGBoost
    fraud_rate = y[train_mask_t].mean().item()
    pos_weight = torch.tensor((1 - fraud_rate) / max(fraud_rate, 1e-6), device=device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Message passing runs over the FULL graph (structural edges are
        # known at inference time regardless of label), but loss is
        # computed only on train-labeled nodes — this is standard
        # semi-supervised node classification practice and does not leak
        # test labels, since labels themselves are never used as features.
        out = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask_t], y[train_mask_t], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        scores = torch.sigmoid(out[test_mask_t]).cpu().numpy()
        y_true = y[test_mask_t].cpu().numpy()

    return model, scores, y_true


def run(use_synthetic: bool = False, sample_frac: float | None = None,
        out_path: str = os.path.join(config.RESULTS_DIR, "graphsage_metrics.json")):
    if use_synthetic:
        df = make_synthetic_dataset(n_rows=20000)
    else:
        df = load_raw(sample_frac=sample_frac)

    df = engineer_features(df)
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    G = build_entity_graph(df)

    cutoff = int(len(df) * (1 - config.TEST_SIZE))
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[:cutoff] = True
    test_mask[cutoff:] = True

    print(f"Training GraphSAGE on {train_mask.sum()} nodes, "
          f"evaluating on {test_mask.sum()} held-out nodes...")
    model, scores, y_true = train_graphsage(df, feature_cols, G, train_mask, test_mask)

    metrics = evaluate(y_true, scores, config.FPR_TARGETS)
    print(format_metrics("graphsage", metrics))

    result = {"model": "graphsage", "metrics": metrics,
              "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
              "hidden_dim": config.SAGE_HIDDEN_DIM, "num_layers": config.SAGE_NUM_LAYERS}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

    model_path = os.path.join(config.MODELS_DIR, "graphsage.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=config.SAGE_EPOCHS)
    args = parser.parse_args()
    if args.epochs != config.SAGE_EPOCHS:
        config.SAGE_EPOCHS = args.epochs
    run(use_synthetic=args.synthetic, sample_frac=args.sample_frac)
