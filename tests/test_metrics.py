import numpy as np

from src.metrics import evaluate, recall_at_fpr


def test_recall_at_fpr_perfect_separation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    assert recall_at_fpr(y_true, y_score, target_fpr=0.0) == 1.0


def test_recall_at_fpr_random_scores_is_bounded():
    rng = np.random.default_rng(0)
    y_true = (rng.random(2000) < 0.1).astype(int)
    y_score = rng.random(2000)
    r = recall_at_fpr(y_true, y_score, target_fpr=0.05)
    assert 0.0 <= r <= 1.0
    # Random scores at 5% FPR should recall roughly 5% of positives, not
    # anywhere close to perfect.
    assert r < 0.5


def test_evaluate_returns_expected_keys():
    rng = np.random.default_rng(1)
    n = 1000
    y_true = (rng.random(n) < 0.035).astype(int)
    y_score = y_true * rng.random(n) * 0.6 + rng.random(n) * 0.4
    metrics = evaluate(y_true, y_score, fpr_targets=(0.01, 0.05))
    assert set(metrics.keys()) == {"pr_auc", "recall@1%fpr", "recall@5%fpr"}
    assert 0.0 <= metrics["pr_auc"] <= 1.0
