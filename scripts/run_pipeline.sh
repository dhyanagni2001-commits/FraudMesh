#!/usr/bin/env bash
# Run the full FraudMesh pipeline end to end: baseline -> graph-augmented
# XGBoost -> GraphSAGE -> case study. Produces results/*.json used in the
# README ablation table.
#
# Usage:
#   ./scripts/run_pipeline.sh            # uses real data/train_transaction.csv
#   ./scripts/run_pipeline.sh --synthetic  # uses generated synthetic data
#
set -euo pipefail
cd "$(dirname "$0")/.."

FLAG="${1:-}"

echo "== Phase 1: XGBoost baseline =="
python3 src/train_baseline.py $FLAG

echo ""
echo "== Phase 2-3: Graph construction + graph-augmented XGBoost =="
python3 src/train_graph_features.py $FLAG

echo ""
echo "== Phase 4: GraphSAGE =="
python3 src/train_graphsage.py $FLAG

echo ""
echo "== Phase 5: Fraud ring case study =="
python3 src/case_study.py $FLAG

echo ""
echo "All results written to results/. See README.md for the ablation table."
