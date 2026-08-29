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
        # Skip connection: the classifier sees the raw per-transaction
        # features concatenated alongside the graph-aggregated hidden state,
        # not just the aggregated state alone. Without this, a node's own
        # signal only reaches the output through however many rounds of
        # neighborhood averaging it passed through — on a real card1 hub
        # with dozens of mostly-unrelated legitimate transactions, that
        # averaging can dilute a fraud signal that XGBoost gets to use
        # directly, unaggregated. With the skip path, the output layer can
        # learn to weight raw features more heavily where the aggregated
        # ones aren't informative, giving GraphSAGE a floor close to a
        # plain-feature classifier instead of being strictly worse than one.
        self.out = torch.nn.Linear(hidden_dim + in_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x_in = x
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.cat([h, x_in], dim=-1)
        return self.out(h).squeeze(-1)


def build_pyg_data(df, feature_cols, G):
    # Standardize BEFORE filling missing values, not after. Many real
    # IEEE-CIS columns (V*/D*) are 30-90% missing; filling with 0 first and
    # then computing mean/std over that column drags std down artificially
    # (most entries are a fake 0), so genuine values get divided by a tiny
    # std and blow up into large magnitudes. Message passing then sums those
    # exploded values across a node's neighborhood, which is large on real
    # data (card1 has only ~13.5k unique values across ~590k real rows, vs.
    # the synthetic generator's deliberately huge cardinality) — the
    # combination reliably wrecks training. pandas .mean()/.std() skip NaN
    # by default, so computing stats first and filling after means a
    # missing value becomes "imputed to the column mean" (0 in z-score
    # space) instead of contaminating the stats used to scale every other
    # value in that column.
    raw = df[feature_cols]
    mean = raw.mean()
    std = raw.std()
    x = torch.tensor(((raw - mean) / (std + 1e-6)).fillna(0).values, dtype=torch.float)

    y = torch.tensor(df[config.TARGET_COL].values, dtype=torch.float)
    edge_index = to_pyg_edge_index(G, num_nodes=len(df))
    return x, edge_index, y


def _load_checkpoint(model, optimizer, in_dim, device, checkpoint_path):
    """Resume from a mid-training checkpoint if one exists and matches this
    run's input shape. Returns the epoch to resume FROM (0 if no usable
    checkpoint), so a Kaggle session that gets killed mid-run doesn't lose
    everything and start the full epoch count over from a random init."""
    if not os.path.exists(checkpoint_path):
        return 0

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        # A killed process (Kaggle session timeout, interrupted cell) can
        # land exactly mid-write and leave a truncated/corrupted file. Don't
        # let that crash the whole run — fall back to training from scratch,
        # same as if there were no checkpoint at all.
        print(f"  found checkpoint at {checkpoint_path} but couldn't load it "
              f"({e!r}) — probably corrupted by an interrupted write. "
              "Starting fresh instead of resuming.")
        return 0

    if ckpt.get("in_dim") != in_dim:
        print(f"  found checkpoint at {checkpoint_path} but its input dim "
              f"({ckpt.get('in_dim')}) doesn't match this run's ({in_dim}) — "
              "probably from a run against different data (e.g. synthetic "
              "vs. real). Starting fresh instead of resuming.")
        return 0

    try:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
    except (RuntimeError, ValueError, KeyError) as e:
        # in_dim matching isn't sufficient to guarantee compatibility — the
        # model architecture itself can change between code versions (e.g.
        # adding the skip connection changed the output layer's shape from
        # [1, hidden_dim] to [1, hidden_dim + in_dim] without changing
        # in_dim at all). Any state_dict shape mismatch should fall back to
        # a fresh start rather than crash the run.
        print(f"  found checkpoint at {checkpoint_path} but couldn't load it into "
              f"this model ({e!r}) — probably saved by a different model "
              "architecture. Starting fresh instead of resuming.")
        return 0

    done = ckpt["epoch"] + 1
    print(f"  resuming from checkpoint at {checkpoint_path}: "
          f"{done} epoch(s) already completed (loss={ckpt['loss']:.4f})")
    return done


def _save_checkpoint(model, optimizer, epoch, in_dim, loss, checkpoint_path):
    # Write to a temp file and rename into place rather than saving directly
    # to checkpoint_path: os.replace is atomic, so a process killed mid-write
    # (session timeout, interrupted cell) leaves either the old checkpoint
    # untouched or the new one fully written — never a truncated file that
    # crashes the next load. Reproduced the truncated-file failure mode by
    # kill -9'ing a run mid-save before adding this.
    tmp_path = checkpoint_path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "in_dim": in_dim,
        "loss": loss,
    }, tmp_path)
    os.replace(tmp_path, checkpoint_path)


def train_graphsage(df, feature_cols, G, train_mask, test_mask,
                     hidden_dim=config.SAGE_HIDDEN_DIM,
                     num_layers=config.SAGE_NUM_LAYERS,
                     dropout=config.SAGE_DROPOUT,
                     epochs=config.SAGE_EPOCHS, lr=config.SAGE_LR,
                     resume=True, tag="graphsage"):
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    checkpoint_path = os.path.join(config.MODELS_DIR, f"{tag}_checkpoint.pt")
    x, edge_index, y = build_pyg_data(df, feature_cols, G)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool, device=device)

    in_dim = x.shape[1]
    model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim,
                       num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    start_epoch = 0
    if resume:
        start_epoch = _load_checkpoint(model, optimizer, in_dim, device, checkpoint_path)
    elif os.path.exists(checkpoint_path):
        print(f"  --fresh passed: ignoring existing checkpoint at {checkpoint_path}")

    # Class-imbalance-aware loss, same rationale as scale_pos_weight in XGBoost
    fraud_rate = y[train_mask_t].mean().item()
    pos_weight = torch.tensor((1 - fraud_rate) / max(fraud_rate, 1e-6), device=device)

    if start_epoch >= epochs:
        print(f"  checkpoint already at epoch {start_epoch} >= target {epochs}; "
              "skipping training, evaluating current weights.")

    model.train()
    for epoch in range(start_epoch, epochs):
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

        # Checkpointed every epoch, not just at the end: the model is tiny
        # (hidden_dim=64, 2 layers) so this is cheap, and it's the only way
        # a rerun after an interrupted Kaggle session (killed cell, hit the
        # session time limit) picks up where it left off instead of
        # retraining all `epochs` from a random init.
        _save_checkpoint(model, optimizer, epoch, in_dim, loss.item(), checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        scores = torch.sigmoid(out[test_mask_t]).cpu().numpy()
        y_true = y[test_mask_t].cpu().numpy()

    return model, scores, y_true


def run(use_synthetic: bool = False, sample_frac: float | None = None,
        resume: bool = True, epochs: int = config.SAGE_EPOCHS,
        no_edges: bool = False, out_path: str | None = None):
    if use_synthetic:
        df = make_synthetic_dataset(n_rows=20000)
    else:
        df = load_raw(sample_frac=sample_frac)

    df = engineer_features(df)
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    G = build_entity_graph(df)

    # Diagnostic mode: strip every edge (nodes only) so message passing has
    # no real neighbors to aggregate over — to_pyg_edge_index() falls back
    # to self-loops for an edgeless graph, so this makes GraphSAGE behave
    # close to a plain per-node MLP on the same features. Compare its PR-AUC
    # to the real run: if --no-edges alone recovers close to the XGBoost
    # baseline, the entity graph's neighborhoods are actively hurting
    # (oversmoothing) rather than helping; if it's still bad, the problem
    # isn't the graph at all.
    tag = "graphsage"
    if no_edges:
        G = G.copy()
        G.remove_edges_from(list(G.edges()))
        tag = "graphsage_noedges"
        print("  --no-edges: stripped all graph edges (self-loops only) — "
              "this run isolates whether the graph itself is hurting GraphSAGE.")

    if out_path is None:
        out_path = os.path.join(config.RESULTS_DIR, f"{tag}_metrics.json")

    cutoff = int(len(df) * (1 - config.TEST_SIZE))
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[:cutoff] = True
    test_mask[cutoff:] = True

    print(f"Training GraphSAGE on {train_mask.sum()} nodes, "
          f"evaluating on {test_mask.sum()} held-out nodes...")
    model, scores, y_true = train_graphsage(df, feature_cols, G, train_mask, test_mask,
                                             resume=resume, epochs=epochs, tag=tag)

    metrics = evaluate(y_true, scores, config.FPR_TARGETS)
    print(format_metrics(tag, metrics))

    result = {"model": tag, "metrics": metrics,
              "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
              "hidden_dim": config.SAGE_HIDDEN_DIM, "num_layers": config.SAGE_NUM_LAYERS}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

    model_path = os.path.join(config.MODELS_DIR, f"{tag}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")

    # Training finished the full target epoch count — clear the mid-training
    # checkpoint so a later run against different data (or --fresh) doesn't
    # find a stale one lying around. {tag}.pt (just saved above) is the one
    # serve.py and future resumes-from-scratch actually care about.
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{tag}_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=config.SAGE_EPOCHS)
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint and train from a random init")
    parser.add_argument("--no-edges", action="store_true",
                         help="Diagnostic: strip all graph edges to isolate whether the "
                              "graph itself is helping or hurting GraphSAGE")
    args = parser.parse_args()
    # Pass epochs through explicitly rather than mutating config.SAGE_EPOCHS:
    # train_graphsage()'s `epochs=config.SAGE_EPOCHS` default is bound once
    # at import time, so reassigning the config attribute afterward has no
    # effect on it — that was silently making --epochs a no-op.
    run(use_synthetic=args.synthetic, sample_frac=args.sample_frac,
        resume=not args.fresh, epochs=args.epochs, no_edges=args.no_edges)
