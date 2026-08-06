"""
Evaluation metrics tailored to imbalanced fraud detection.

Plain accuracy and ROC-AUC are both misleading at ~3.5% positive rate.
We report:
  - PR-AUC (average precision): the headline number
  - recall@FPR: "if the review team can only tolerate an X% false-positive
    rate on legitimate transactions, what fraction of fraud do we catch?"
    This maps directly to a fraud-ops review-queue capacity constraint.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> float:
    """Return the max recall achievable at or below target_fpr."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    return float(tpr[idx])


def evaluate(y_true, y_score, fpr_targets=(0.01, 0.05)) -> dict:
    """Compute the full metric bundle used across every model in this repo."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    metrics = {"pr_auc": float(average_precision_score(y_true, y_score))}
    for target in fpr_targets:
        pct = int(target * 100)
        metrics[f"recall@{pct}%fpr"] = recall_at_fpr(y_true, y_score, target)
    return metrics


def format_metrics(name: str, metrics: dict) -> str:
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return f"[{name}] " + ", ".join(parts)


if __name__ == "__main__":
    # Quick self-test with synthetic scores
    rng = np.random.default_rng(0)
    n = 5000
    y = (rng.random(n) < 0.035).astype(int)
    scores = y * rng.random(n) * 0.6 + rng.random(n) * 0.4  # weak but real signal
    m = evaluate(y, scores, fpr_targets=(0.01, 0.05))
    print(format_metrics("self_test", m))
