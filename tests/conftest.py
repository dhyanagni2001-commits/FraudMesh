import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from src.data_prep import engineer_features, make_synthetic_dataset  # noqa: E402


@pytest.fixture
def synthetic_df():
    """Small synthetic dataset, engineered, for fast unit tests."""
    df = make_synthetic_dataset(n_rows=500, seed=0)
    return engineer_features(df)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Redirect config.MODELS_DIR / config.RESULTS_DIR to a temp dir so
    tests never write into (or depend on) the real models/ or results/."""
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    models_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(config, "RESULTS_DIR", str(results_dir))
    return {"models_dir": models_dir, "results_dir": results_dir}
