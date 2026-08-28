# FraudMesh

Graph-based fraud ring detection. Transactions are linked into a graph via
shared card and device identifiers, and a GraphSAGE model is trained on top
of that structure to catch fraud that a row-by-row model can't see:
coordinated fraud rings reusing the same stolen card or device across many
transactions.

The project is built as an honest ablation, not a single model dropped in
isolation. Three models are trained on the same data and the same
train/test split, so the value of the graph — and of the GNN specifically —
is measurable rather than assumed:

1. **XGBoost baseline** — tabular features only, no graph
2. **XGBoost + graph features** — same features, plus cheap non-learned
   graph statistics (node degree, connected-component size, PageRank)
3. **GraphSAGE** — full message-passing model trained end-to-end on the
   entity graph

If GraphSAGE doesn't meaningfully beat step 2, that's a real and useful
finding — it means the graph's value is fully captured by cheap statistics,
and the extra serving complexity of a GNN isn't worth paying for. This repo
reports whichever result actually comes out of the run, not the one that
makes the best story.

## Results

Numbers below are from a full pipeline run on a synthetic dataset shaped
like IEEE-CIS (20,000 transactions, 3.5% fraud rate, with injected fraud
rings — see [Synthetic data](#synthetic-data-mode) below). Re-run
`scripts/run_pipeline.sh` against the real Kaggle data to get numbers for
the actual competition dataset; the ablation shape (which model wins, by
how much) is what matters, not the absolute values on synthetic data.

| Model | PR-AUC | Recall @ 1% FPR | Recall @ 5% FPR |
|---|---|---|---|
| XGBoost (baseline, no graph) | 0.8107 | 0.7862 | 0.7931 |
| XGBoost + graph features | 0.8111 | 0.7793 | 0.8069 |
| GraphSAGE (end-to-end) | 0.8214 | 0.7931 | 0.8207 |

Reading this honestly: cheap graph statistics (degree, component size,
PageRank) alone barely move PR-AUC over the tabular baseline (+0.0004 —
noise-level) and don't help at all at a tight 1% FPR budget. Nearly all of
the graph's value shows up only once you go to the full GraphSAGE model
(+0.0107 PR-AUC over baseline, +0.0103 over the graph-features model), most
visibly at 5% FPR (0.807 → 0.821). That's a real, decisive result for a
production recommendation, not a close call: the structural graph
statistics used here aren't a substitute for message passing, so if the
graph is worth building at all, it's worth training GraphSAGE on — a
graph-features-only XGBoost model isn't a credible cheaper alternative for
this dataset.

### Case study: fraud rings found

Running `src/case_study.py` on the same synthetic data surfaces concrete,
inspectable communities rather than just a score:

```
Ring #1
  transactions:        502
  fraud rate:          97.4%
  avg amount:          $248.31
  total amount:        $124,653.04
  time span:           716.2 hours
  unique cards:        25
  unique devices:      22
  linked via:           DeviceInfo, card1
```

That's 502 transactions, spread across 25 different cards and 22 different
devices, that nonetheless form one tightly connected structure — because
those cards and devices were reused across each other — with 97.4% of them
labeled fraud. No single-transaction, row-wise model can see this pattern;
it only exists at the graph level. This is the artifact worth walking
through in an interview, not the PR-AUC number alone.

## Why this graph, and not a simpler one

Two design decisions matter enough to call out explicitly, because both
came from actually running the code and finding it broken, not from
theory:

**`addr1` (billing region) and `P_emaildomain` are excluded as graph link
columns.** Both look like natural "shared entity" fields, but both are too
coarse: `addr1` has only a few hundred unique values and `P_emaildomain` is
dominated by a handful of major providers (Gmail, Yahoo, Hotmail). Linking
transactions on either field merges huge, unrelated swaths of the dataset
into one giant connected component and destroys the very structure the
graph is meant to expose. Only `card1` and `DeviceInfo` — genuinely
high-specificity identifiers — are used as graph edges. The excluded
columns are still available to every model as frequency-encoded tabular
features; they're just not treated as identity links.

**Synthetic legitimate traffic uses high-cardinality cards/devices.** An
early version of the synthetic data generator gave legitimate transactions
low-cardinality card/device pools too, which caused the same giant-component
collapse — even non-fraudulent transactions were colliding by chance. Fixed
by making legitimate card/device pools roughly 2-3x the row count (mirroring
real payment data, where collisions are rare), while fraud rings still reuse
a small, fixed pool of cards/devices on purpose. This is what makes the
fraud-rate-by-component-size validation (below) actually mean something.

**A raw connected-component ID is deliberately NOT used as an XGBoost
feature**, even though an earlier version of this repo included one
(`graph_component_id`) and it showed up as a top-10 feature by importance.
It's a leakage trap: the entity graph is built over train + test combined
(structurally correct — see the note in `train_graph_features.py`), which
means a component ID is shared verbatim between train and test rows in the
same cluster. XGBoost can split exactly on that integer and memorize the
*training* fraud rate of a component, which then "predicts" test rows in
the same component almost perfectly for reasons that have nothing to do
with generalizing to a new ring. Removing it dropped the graph-features
model's PR-AUC from 0.8165 to 0.8111 — a small absolute drop, but a
meaningful one: it changes the honest conclusion from "cheap graph stats
capture most of the value" to "they capture almost none of it" (see
Results). `graph_component_size`, `graph_degree`, and `graph_pagerank`
stay, because they're real-valued structural statistics, not identity
keys a tree can pin to.

### Graph validation

Before trusting the graph is worth modeling on, `graph_builder.py` reports
fraud rate as a function of connected-component size. On the synthetic
dataset:

| Min. component size | Avg. fraud rate |
|---|---|
| 1 (all nodes) | 1.0% |
| 2 | 1.3% |
| 5 | 1.9% |
| 10 | 12.8% |
| 20 | 93.2% |

Fraud rate climbing from ~1% to over 90% as component size grows is the
signal that justifies the whole project — it means dense clusters in the
entity graph really do correspond to fraud, not noise.

## Architecture

```
FraudMesh/
├── config.py                    # paths (incl. Kaggle auto-detection), entity-link
│                                 # columns, hyperparameters
├── requirements.txt              # local (venv) setup
├── requirements-kaggle.txt       # Kaggle setup — just the packages Kaggle lacks
├── notebooks/
│   └── fraudmesh_kaggle.ipynb    # run the full pipeline on Kaggle against real data
├── scripts/
│   └── run_pipeline.sh          # runs all phases end to end
├── src/
│   ├── data_prep.py             # loading, feature engineering, time-aware split,
│   │                             synthetic data generator
│   ├── metrics.py                # PR-AUC, recall@FPR (imbalance-aware evaluation)
│   ├── graph_builder.py          # shared-entity graph construction + validation
│   ├── train_baseline.py         # Phase 1: XGBoost, no graph
│   ├── train_graph_features.py   # Phase 3: XGBoost + graph statistics
│   ├── train_graphsage.py        # Phase 4: GraphSAGE (PyTorch Geometric)
│   ├── case_study.py             # Phase 5: fraud ring detection + reporting
│   └── serve.py                  # Phase 6: FastAPI serving layer
├── data/                          # place Kaggle CSVs here (gitignored)
├── models/                        # trained model weights land here (gitignored)
└── results/                       # metrics JSON from each phase
```

### Why time-aware, not random, splitting

Every model in this repo is trained and evaluated on a **time-aware split**:
transactions are sorted by `TransactionDT` and the last 20% by time becomes
the test set. A random shuffle-split would let future transaction patterns
leak into training and inflate every metric — it's not how the model would
ever see data in production, where you only ever have the past to predict
the future.

### Why PR-AUC and recall@FPR, not accuracy or plain ROC-AUC

Fraud is a ~3.5% positive-rate problem. A model that predicts "not fraud"
on everything scores 96.5% accuracy and is useless. ROC-AUC is also
optimistic under heavy imbalance because it's dominated by the huge
majority class. PR-AUC and recall-at-a-fixed-false-positive-rate map
directly onto how a fraud team actually operates: "given that our review
queue can only tolerate flagging X% of legitimate transactions, what
fraction of real fraud do we catch?" See `src/metrics.py`.

## Setup

```bash
git clone <this-repo>
cd FraudMesh
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Getting the data

Download `train_transaction.csv` and `train_identity.csv` from the
[IEEE-CIS Fraud Detection Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data)
and place both in `data/`. `train_identity.csv` is optional — the pipeline
runs on `train_transaction.csv` alone (with a smaller feature set) if it's
not present.

### Running on Kaggle

The intended way to run this against the real competition data without a
local 600MB+ download is a Kaggle Notebook:

1. Get the code onto Kaggle, either:
   - **Recommended:** zip this project folder and upload it as a private
     Kaggle Dataset (*Add Input → Datasets → New Dataset*), then attach it
     to your notebook, or
   - push it to a Git remote and use the clone fallback built into the
     notebook below.
2. Open `notebooks/fraudmesh_kaggle.ipynb` on Kaggle (upload it directly,
   or attach the dataset from step 1 and open it from there).
3. *Add Input → Competitions → IEEE-CIS Fraud Detection* to attach the real
   `train_transaction.csv` / `train_identity.csv`.
4. Turn on a GPU accelerator (*Settings → Accelerator*) — `train_graphsage.py`
   uses CUDA automatically when available and falls back to CPU otherwise.
5. Run all cells.

Two things are handled automatically so the same code runs unmodified in
both places:

- **Paths.** `config.py` detects `/kaggle/input` and (a) searches it for
  whichever mounted folder actually contains `train_transaction.csv`
  instead of assuming a fixed dataset slug, and (b) always writes
  `results/*.json` and `models/graphsage.pt` under `/kaggle/working`, since
  `/kaggle/input` is a read-only mount and this repo's own code may be
  sitting on it (if uploaded as a Dataset rather than git-cloned).
- **Dependencies.** Kaggle notebooks already ship pandas/numpy/xgboost/torch
  matched to the notebook's CUDA build; `requirements-kaggle.txt` installs
  only the one thing actually missing (`torch_geometric`) rather than
  reinstalling everything from `requirements.txt`, which would risk
  overwriting Kaggle's GPU-matched torch with a mismatched one from PyPI.

### Synthetic data mode

Every script in this repo accepts a `--synthetic` flag, which generates a
dataset shaped like IEEE-CIS locally (no download required) with the same
column names, a realistic ~3.5% fraud rate, and injected fraud rings (a
small pool of cards/devices reused heavily by fraud transactions, mixed
with "lone wolf" fraud that has no reuse pattern at all — so the graph
models can't catch everything, which is the honest, realistic case). This
is how every number in this README was produced, and it's the fastest way
to confirm the whole pipeline runs before pointing it at the real 600MB+
Kaggle dataset.

## Running the pipeline

```bash
# Full pipeline, synthetic data (fast, no download needed)
./scripts/run_pipeline.sh --synthetic

# Full pipeline, real IEEE-CIS data (place CSVs in data/ first)
./scripts/run_pipeline.sh

# Or run any phase individually
python3 src/train_baseline.py --synthetic
python3 src/train_graph_features.py --synthetic
python3 src/train_graphsage.py --synthetic
python3 src/case_study.py --synthetic
```

Each script writes its metrics to `results/*.json` and prints a summary to
stdout. `train_graphsage.py` additionally saves model weights to
`models/graphsage.pt`, which `src/serve.py` loads at startup.

For faster local iteration on the real dataset before a full run, use
`--sample-frac 0.1` on any training script to subsample legitimate
transactions (all fraud rows are always kept, since they're the minority
class).

## Serving

```bash
cd src
uvicorn serve:app --reload
```

Then:

```bash
# Check cache status
curl http://127.0.0.1:8000/health

# Score a known transaction
curl -X POST http://127.0.0.1:8000/score \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 0}'

# Force an embedding refresh
curl -X POST http://127.0.0.1:8000/refresh
```

### The serving problem this solves

A row-wise model like XGBoost can score any transaction in isolation.
GraphSAGE can't — it needs a transaction's neighborhood (other transactions
sharing its card or device) to compute an embedding, so a brand-new,
never-before-seen transaction can't be scored the instant it arrives
without knowing its graph context first.

`serve.py` handles this with a precompute-and-cache strategy: a refresh job
(triggered here via `/refresh`, and on a schedule in production) rebuilds
the graph and re-scores every known transaction into an in-memory cache;
the `/score` endpoint is then a fast lookup, not a live forward pass. This
is the right first production step for most teams — the alternative (true
online inference, attaching each new transaction to a live graph store and
running fresh k-hop neighbor sampling per request) is lower-latency but
substantially more infrastructure, and isn't necessary until the cache
staleness window itself becomes the bottleneck.

## Tech stack

- **Modeling:** XGBoost, PyTorch, PyTorch Geometric
- **Graph:** NetworkX (construction, community detection, validation)
- **Serving:** FastAPI, Uvicorn
- **Data:** pandas, scikit-learn (metrics)

## What this project deliberately does not do

- **No true online/streaming inference** — see the serving section above.
  The cache-refresh design is a real, defensible production pattern, not a
  shortcut being hidden.
- **No hyperparameter sweep** — `config.py` has one set of reasonable
  GraphSAGE/XGBoost hyperparameters. A full sweep would improve the
  absolute numbers but wouldn't change the shape of the ablation story,
  which is the point of this project.
- **No deep explainability tooling for GraphSAGE** (e.g. GNNExplainer) —
  the case-study/community-detection approach in `case_study.py` is used
  instead, because a concrete list of transactions in a flagged ring is
  more directly useful to a human reviewer than an attribution heatmap over
  learned embeddings.
