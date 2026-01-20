"""Invariance tests for proxy metrics."""

from __future__ import annotations

from eval_harness.experiments import _apply_invariance_pair, _evaluate_proxy
from eval_harness.io import split_train_test
from eval_harness.preprocess import fit_preprocess
from eval_harness.generators import BootstrapGenerator


def test_invariance_transforms(toy_df, toy_schema, test_config):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    preprocess = fit_preprocess(train_df, toy_schema)

    gen = BootstrapGenerator()
    gen.fit(train_df, toy_schema, seed=0)
    synth_df = gen.sample(len(train_df), seed=0)

    _, base_summary = _evaluate_proxy(
        train_df,
        test_df,
        synth_df,
        toy_schema,
        preprocess,
        model_names=test_config["utility"]["models"],
        seeds=test_config["generators"]["seeds"],
        ndcg_ks=test_config["proxy_metrics"]["ndcg_ks"],
        rbo_ps=test_config["proxy_metrics"]["rbo_ps"],
        jaccard_ks=test_config["proxy_metrics"]["jaccard_ks"],
        overlap_ks=test_config["proxy_metrics"]["overlap_ks"],
        scoring=None,
        n_repeats=test_config["preprocess"]["n_repeats_importance"],
    )

    for kind in ["column_reorder", "categorical_bijection", "numeric_affine"]:
        r_train, r_test, s_df = _apply_invariance_pair(
            train_df,
            test_df,
            synth_df,
            toy_schema,
            kind,
            seed=123,
        )
        preprocess_inv = fit_preprocess(r_train, toy_schema)
        _, summary = _evaluate_proxy(
            r_train,
            r_test,
            s_df,
            toy_schema,
            preprocess_inv,
            model_names=test_config["utility"]["models"],
            seeds=test_config["generators"]["seeds"],
            ndcg_ks=test_config["proxy_metrics"]["ndcg_ks"],
            rbo_ps=test_config["proxy_metrics"]["rbo_ps"],
            jaccard_ks=test_config["proxy_metrics"]["jaccard_ks"],
            overlap_ks=test_config["proxy_metrics"]["overlap_ks"],
            scoring=None,
            n_repeats=test_config["preprocess"]["n_repeats_importance"],
        )
        deltas = [abs(base_summary[m]["median"] - summary[m]["median"]) for m in base_summary]
        assert max(deltas) <= test_config["stats"]["invariance_epsilon"]
