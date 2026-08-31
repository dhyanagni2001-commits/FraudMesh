"""
Phase 6: Serving layer for the trained GraphSAGE model.

The core serving challenge with a GNN, unlike a row-wise model, is that you
cannot score a transaction in isolation — GraphSAGE needs the transaction's
neighborhood (other transactions sharing its card/device) to compute an
embedding. Two practical strategies exist:

1. Precompute + cache: periodically re-run the graph + GraphSAGE forward
   pass over the full recent transaction window, cache node embeddings, and
   at request time just do a cheap lookup + small classifier head. This is
   what's implemented below — a background refresh job populates an
   in-memory cache, and the API serves off the most recent cache.
2. True online inference: attach a brand-new transaction to the live graph
   as it arrives and run a fresh forward pass. Lower latency to "ring
   awareness" but much higher engineering cost (live graph store, k-hop
   neighbor sampling online) and out of scope for this project.

This app implements (1), which is the right first production step for most
teams: a `refresh_embeddings()` job runs on a schedule (simulated here via
a manual /refresh endpoint) and scoring is a fast in-memory operation.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import engineer_features, get_feature_columns, make_synthetic_dataset
from src.graph_builder import build_entity_graph, to_pyg_edge_index
from src.train_graphsage import GraphSAGE, build_pyg_data

app = FastAPI(title="FraudMesh Serving API", version="0.1.0")

# ---- In-memory cache, populated by refresh_embeddings() ----
_cache = {
    "model": None,
    "model_trained": False,
    "feature_cols": None,
    "df": None,
    "embeddings": None,   # not exposed separately here; we cache final scores
    "scores": None,       # {transaction_index: fraud_probability}
    "last_refresh": None,
}


class TransactionQuery(BaseModel):
    transaction_id: int


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    cache_age_seconds: float
    note: Optional[str] = None


def _load_model(feature_dim: int) -> tuple[GraphSAGE, bool]:
    """Returns (model, is_trained). is_trained=False means no weights file
    was found (or it didn't load cleanly), so the model is at its random
    init — every score it produces is meaningless noise, not a real
    prediction. Without tracking this explicitly, a caller has no way to
    tell "0.0043 fraud probability" apart from "this model was never
    trained" — both look like a normal, confident-looking response."""
    model = GraphSAGE(in_dim=feature_dim, hidden_dim=config.SAGE_HIDDEN_DIM,
                       num_layers=config.SAGE_NUM_LAYERS, dropout=config.SAGE_DROPOUT)
    model_path = os.path.join(config.MODELS_DIR, "graphsage.pt")
    is_trained = False
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            is_trained = True
        except (RuntimeError, ValueError) as e:
            print(f"WARNING: found {model_path} but couldn't load it ({e!r}) — "
                  "probably saved by a different model architecture or feature "
                  "set. Serving with an untrained (random-init) model instead "
                  "of crashing; scores will be meaningless until this is fixed.")
    else:
        print(f"WARNING: no trained model found at {model_path} — serving with "
              "an untrained (random-init) model. Run train_graphsage.py first.")
    model.eval()
    return model, is_trained


def _default_use_synthetic() -> bool:
    """Match whatever the currently-saved model was most likely trained on.
    An explicit FRAUDMESH_SYNTHETIC env var always wins; otherwise default
    to real data if it's present, since a model trained on real IEEE-CIS
    features has a different input dimension than one trained on synthetic
    data and loading it against the wrong feature set will fail."""
    env = os.environ.get("FRAUDMESH_SYNTHETIC")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no")
    return not os.path.exists(config.TRANSACTION_CSV)


def refresh_embeddings(use_synthetic: Optional[bool] = None):
    """Rebuild the graph and re-score every known transaction. In production
    this would run on a schedule (e.g. every 5-15 minutes) against the
    latest rolling window of transactions, not the whole history — scoped
    down here for a runnable local demo."""
    if use_synthetic is None:
        use_synthetic = _default_use_synthetic()

    if use_synthetic:
        df = make_synthetic_dataset(n_rows=5000)
    else:
        from src.data_prep import load_raw
        df = load_raw()

    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    G = build_entity_graph(df)

    x, edge_index, _ = build_pyg_data(df, feature_cols, G)
    model, is_trained = _load_model(feature_dim=x.shape[1])

    with torch.no_grad():
        out = model(x, edge_index)
        scores = torch.sigmoid(out).numpy()

    _cache["model"] = model
    _cache["model_trained"] = is_trained
    _cache["feature_cols"] = feature_cols
    _cache["df"] = df
    _cache["scores"] = {int(tid): float(s) for tid, s in
                         zip(df["TransactionID"], scores)}
    _cache["last_refresh"] = time.time()


@app.on_event("startup")
def on_startup():
    refresh_embeddings()


@app.post("/refresh")
def manual_refresh():
    """Trigger an out-of-schedule embedding refresh."""
    refresh_embeddings()
    return {"status": "refreshed", "n_transactions": len(_cache["scores"])}


@app.post("/score", response_model=ScoreResponse)
def score_transaction(query: TransactionQuery):
    if _cache["scores"] is None:
        raise HTTPException(status_code=503, detail="Cache not yet populated, call /refresh")

    score = _cache["scores"].get(query.transaction_id)
    if score is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {query.transaction_id} not found in the cached "
                "graph. New transactions must go through /refresh (or, in "
                "production, wait for the next scheduled refresh) before "
                "they can be scored, since GraphSAGE needs their entity "
                "neighborhood to compute an embedding."
            ),
        )

    age = time.time() - _cache["last_refresh"]
    note = None
    if not _cache["model_trained"]:
        note = ("WARNING: no trained model was found — this score comes from "
                 "an untrained (random-init) model and is meaningless. Run "
                 "train_graphsage.py, then call /refresh.")
    elif age > 900:
        note = "Cache is over 15 minutes old — consider calling /refresh."

    return ScoreResponse(
        transaction_id=query.transaction_id,
        fraud_probability=score,
        cache_age_seconds=age,
        note=note,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cache_populated": _cache["scores"] is not None,
        "model_trained": _cache["model_trained"],
        "n_cached_transactions": len(_cache["scores"]) if _cache["scores"] else 0,
        "cache_age_seconds": (
            time.time() - _cache["last_refresh"] if _cache["last_refresh"] else None
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=False)
