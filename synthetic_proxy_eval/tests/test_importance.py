"""Tests for feature importance and ranking."""

from __future__ import annotations

from eval_harness.importance import compute_importance_rankings, rank_importances
from eval_harness.preprocess import fit_preprocess
from eval_harness.io import split_train_test


def test_rank_importances_ties():
    importance = {"b": 1.0, "a": 1.0, "c": 0.5}
    ranking = rank_importances(importance)
    assert ranking.order == ["a", "b", "c"]
    assert ranking.ranks["a"] == ranking.ranks["b"]


def test_compute_importance_rankings(toy_df, toy_schema):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    preprocess = fit_preprocess(train_df, toy_schema)
    rankings = compute_importance_rankings(
        train_df,
        test_df,
        preprocess,
        toy_schema,
        model_names=["logreg"],
        seeds=[0],
        scoring=None,
        n_repeats=2,
    )
    assert ("logreg", 0) in rankings
    ranking = rankings[("logreg", 0)]
    assert set(ranking.order) == set(toy_schema.features)
