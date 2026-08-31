"""
Phase 6: Serving layer for the trained GraphSAGE model.

The core serving challenge with a GNN, unlike a row-wise model, is that you
cannot score a transaction in isolation — GraphSAGE needs the transaction's
neighborhood (other transactions sharing its card/device) to compute an
embedding. Two practical strategies exist:

1. Precompute + cache: periodically re-run the graph + GraphSAGE forward
   pass over the full recent transaction window, cache node embeddings, and
   at request time just do a cheap lookup + small classifier head. This is
   what's implemented below — a background scheduler refreshes an in-memory
   cache on a timer, and the API serves off the most recent cache.
2. True online inference: attach a brand-new transaction to the live graph
   as it arrives and run a fresh forward pass. Lower latency to "ring
   awareness" but much higher engineering cost (live graph store, k-hop
   neighbor sampling online) and out of scope for this project.

This app implements (1), which is the right first production step for most
teams: a `refresh_embeddings()` job runs on a schedule (an APScheduler
background job, `FRAUDMESH_REFRESH_INTERVAL_SECONDS` apart) and scoring is
a fast in-memory operation. `/refresh` remains available for an on-demand,
out-of-schedule refresh.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import torch
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_prep import engineer_features, get_feature_columns, make_synthetic_dataset
from src.graph_builder import build_entity_graph
from src.logging_config import configure_logging
from src.train_graphsage import GraphSAGE, build_pyg_data

configure_logging()
logger = logging.getLogger(__name__)

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

_scheduler = BackgroundScheduler()


class TransactionQuery(BaseModel):
    transaction_id: int


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    cache_age_seconds: float
    note: str | None = None


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
            logger.warning(
                "found %s but couldn't load it (%r) — probably saved by a "
                "different model architecture or feature set. Serving with "
                "an untrained (random-init) model instead of crashing; "
                "scores will be meaningless until this is fixed.",
                model_path, e,
            )
    else:
        logger.warning(
            "no trained model found at %s — serving with an untrained "
            "(random-init) model. Run train_graphsage.py first.", model_path,
        )
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


def refresh_embeddings(use_synthetic: bool | None = None):
    """Rebuild the graph and re-score every known transaction. Runs on a
    schedule via the background scheduler (see lifespan() below) against
    the latest rolling window of transactions in a real deployment — scoped
    down here for a runnable local demo (whole history, not a window)."""
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


def _safe_refresh():
    """Wraps refresh_embeddings() so a bad refresh (missing data, a
    corrupted checkpoint, transient I/O error) logs and leaves the previous
    cache in place instead of crashing the process (at startup) or killing
    the background scheduler thread (on a scheduled tick)."""
    try:
        refresh_embeddings()
        logger.info("Embeddings refreshed: %d transactions cached",
                     len(_cache["scores"] or {}))
    except Exception:
        logger.exception(
            "refresh_embeddings() failed; serving from the previous cache "
            "(or an empty one, if this was the first attempt)"
        )


def _auto_refresh_enabled() -> bool:
    return os.environ.get("FRAUDMESH_AUTO_REFRESH", "true").strip().lower() not in (
        "0", "false", "no",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _safe_refresh()
    if _auto_refresh_enabled():
        interval = int(os.environ.get("FRAUDMESH_REFRESH_INTERVAL_SECONDS", "900"))
        _scheduler.add_job(_safe_refresh, "interval", seconds=interval,
                            id="refresh_embeddings", replace_existing=True)
        _scheduler.start()
        logger.info("Background auto-refresh enabled: every %ds", interval)
    else:
        logger.info("Background auto-refresh disabled (FRAUDMESH_AUTO_REFRESH=false)")
    yield
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


app = FastAPI(title="FraudMesh Serving API", version="0.1.0", lifespan=lifespan)

_cors_origins_env = os.environ.get("FRAUDMESH_CORS_ORIGINS", "*").strip()
if _cors_origins_env == "*":
    _cors_origins = ["*"]
    _cors_allow_credentials = False  # spec disallows credentials with a wildcard origin
else:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    _cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """Demo-grade shared-secret auth, not a substitute for a real gateway/
    IdP. Disabled entirely (no auth) unless FRAUDMESH_API_KEY is set —
    keeps local/CI/demo usage frictionless while still showing the pattern
    a real deployment would put behind a proper API gateway."""
    expected = os.environ.get("FRAUDMESH_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


@app.post("/refresh", dependencies=[Depends(require_api_key)])
def manual_refresh():
    """Trigger an out-of-schedule embedding refresh."""
    try:
        refresh_embeddings()
    except Exception as e:
        logger.exception("Manual /refresh failed")
        raise HTTPException(status_code=500, detail=f"Refresh failed: {e!r}") from e
    return {"status": "refreshed", "n_transactions": len(_cache["scores"])}


@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score_transaction(query: TransactionQuery):
    if _cache["scores"] is None:
        raise HTTPException(status_code=503, detail="Cache not yet populated, call /refresh")

    score = _cache["scores"].get(query.transaction_id)
    if score is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {query.transaction_id} not found in the cached "
                "graph. New transactions must go through /refresh (or wait "
                "for the next scheduled refresh) before they can be scored, "
                "since GraphSAGE needs their entity neighborhood to compute "
                "an embedding."
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


def main():
    import uvicorn
    host = os.environ.get("FRAUDMESH_HOST", "0.0.0.0")
    port = int(os.environ.get("FRAUDMESH_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
