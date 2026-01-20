"""Tests for preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from eval_harness.preprocess import fit_preprocess, transform_features
from eval_harness.io import split_train_test


def test_preprocess_no_leakage(toy_df, toy_schema):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    test_df = test_df.copy()
    test_df.loc[test_df.index[:2], "smoker"] = "unknown"

    preprocess = fit_preprocess(train_df, toy_schema)
    X_test = transform_features(preprocess, test_df, toy_schema)

    assert X_test.shape[1] == len(preprocess.feature_names)
    assert len(preprocess.feature_names) == len(preprocess.feature_group)

    # Unseen categories should not create new columns
    assert not any("unknown" in name for name in preprocess.feature_names)


def test_feature_group_mapping(toy_df, toy_schema):
    preprocess = fit_preprocess(toy_df, toy_schema)
    feature_group = preprocess.feature_group
    assert set(feature_group) == set(toy_schema.features)
