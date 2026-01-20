"""Pytest fixtures for evaluation harness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval_harness.io import infer_schema


@pytest.fixture(scope="session")
def toy_df() -> pd.DataFrame:
    """Create a deterministic toy dataset with mixed types."""
    rng = np.random.default_rng(0)
    n = 200
    age = rng.integers(20, 80, size=n)
    bmi = rng.normal(25, 4, size=n)
    sex = rng.choice(["M", "F"], size=n)
    smoker = rng.choice(["yes", "no"], size=n, p=[0.2, 0.8])
    risk = bmi + (sex == "M") * 1.5 + (smoker == "yes") * 2.0 + rng.normal(0, 1, size=n)
    target = (risk > np.median(risk)).astype(int)
    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "sex": sex,
            "smoker": smoker,
            "Class": target,
        }
    )
    return df


@pytest.fixture(scope="session")
def toy_schema(toy_df) -> object:
    """Schema inferred from toy dataset."""
    return infer_schema(toy_df, target="Class")


@pytest.fixture(scope="session")
def test_config() -> dict:
    """Minimal config for test runs."""
    return {
        "seed": 123,
        "split": {"test_size": 0.2, "stratify": True},
        "preprocess": {"n_repeats_importance": 2, "importance_scoring": "auto"},
        "generators": {"seeds": [0, 1], "sizes": [1.0], "output_dir": "synthetic"},
        "utility": {"models": ["logreg", "rf"], "eps": 1.0e-8, "task_type": "auto"},
        "proxy_metrics": {
            "ndcg_ks": [5, 10],
            "rbo_ps": [0.9],
            "jaccard_ks": [5, 10],
            "overlap_ks": [5, 10],
        },
        "fidelity": {"propensity": {"test_size": 0.3, "max_iter": 200}},
        "stats": {
            "bootstrap": {"n_resamples": 50, "seed": 123},
            "acceptable_delta_u": 0.02,
            "precision_target": 0.95,
            "invariance_epsilon": 0.05,
            "sensitivity_lambdas": [0.0, 0.5, 1.0],
        },
        "reporting": {"save_plots": False},
    }
