"""Tests for fidelity metrics."""

from __future__ import annotations

from eval_harness.fidelity_metrics import compute_fidelity_metrics
from eval_harness.preprocess import fit_preprocess
from eval_harness.io import split_train_test


def test_fidelity_identity(toy_df, toy_schema):
    train_df, _ = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    preprocess = fit_preprocess(train_df, toy_schema)
    metrics = compute_fidelity_metrics(
        train_df,
        train_df.copy(),
        toy_schema,
        preprocess,
        seed=123,
        test_size=0.3,
        max_iter=200,
    )
    assert metrics["wasserstein_numeric"] < 1e-6
    assert metrics["jsd_categorical"] < 1e-6
    assert metrics["correlation_distance"] < 1e-6
    assert 0.35 <= metrics["propensity_auc"] <= 0.65
