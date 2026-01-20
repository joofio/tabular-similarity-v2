"""Tests for utility and delta-U calculations."""

from __future__ import annotations

import numpy as np

from eval_harness.preprocess import fit_preprocess
from eval_harness.io import split_train_test
from eval_harness.utility import aggregate_delta_u, compute_delta_u, compute_model_metrics


def test_delta_u_identical_vs_degraded(toy_df, toy_schema):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    preprocess = fit_preprocess(train_df, toy_schema)

    base_metrics = compute_model_metrics(
        train_df,
        test_df,
        preprocess,
        toy_schema,
        model_names=["logreg"],
        seeds=[0],
    )

    synth_metrics = compute_model_metrics(
        train_df.copy(),
        test_df,
        preprocess,
        toy_schema,
        model_names=["logreg"],
        seeds=[0],
    )

    deltas = compute_delta_u(
        base_metrics[("logreg", 0)],
        synth_metrics[("logreg", 0)],
        toy_schema.task_type,
        eps=1.0e-8,
    )
    median, _ = aggregate_delta_u(list(deltas.values()))
    assert abs(median) < 0.05

    degraded = train_df.copy()
    degraded["Class"] = np.random.default_rng(0).permutation(degraded["Class"].to_numpy())
    degraded_metrics = compute_model_metrics(
        degraded,
        test_df,
        preprocess,
        toy_schema,
        model_names=["logreg"],
        seeds=[0],
    )
    degraded_deltas = compute_delta_u(
        base_metrics[("logreg", 0)],
        degraded_metrics[("logreg", 0)],
        toy_schema.task_type,
        eps=1.0e-8,
    )
    degraded_median, _ = aggregate_delta_u(list(degraded_deltas.values()))
    assert degraded_median > median
