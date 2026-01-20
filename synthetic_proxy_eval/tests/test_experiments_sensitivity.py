"""Sensitivity tests for degradation effects."""

from __future__ import annotations

from eval_harness.experiments import _apply_degradation, _evaluate_dataset
from eval_harness.generators import SyntheticDataset
from eval_harness.io import split_train_test
from eval_harness.preprocess import fit_preprocess
from eval_harness.utility import compute_model_metrics


def test_sensitivity_monotone(toy_df, toy_schema, test_config):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    preprocess = fit_preprocess(train_df, toy_schema)
    base_utilities = compute_model_metrics(
        train_df,
        test_df,
        preprocess,
        toy_schema,
        model_names=test_config["utility"]["models"],
        seeds=test_config["generators"]["seeds"],
    )

    lambdas = test_config["stats"]["sensitivity_lambdas"]
    deltas = []
    metric_values = []
    for lam in lambdas:
        degraded = _apply_degradation(train_df, toy_schema, "gaussian_noise", lam, seed=123)
        synth = SyntheticDataset(
            name=f"deg_{lam}",
            generator="degradation",
            seed=123,
            size_multiplier=1.0,
            df=degraded,
            path="",
        )
        summary, _, _, _ = _evaluate_dataset(
            synth,
            train_df,
            test_df,
            toy_schema,
            preprocess,
            test_config,
            base_utilities,
        )
        deltas.append(summary["delta_u_median"])
        metric_values.append(summary["proxy_ndcg@5_median"])

    assert all(x <= y + 0.01 for x, y in zip(deltas, deltas[1:]))
    assert all(x >= y - 0.01 for x, y in zip(metric_values, metric_values[1:]))
