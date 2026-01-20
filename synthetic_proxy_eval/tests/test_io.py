"""Tests for IO utilities."""

from __future__ import annotations

import numpy as np

from eval_harness.io import infer_schema, load_dataset, split_train_test


def test_load_dataset_and_schema(tmp_path, toy_df):
    path = tmp_path / "toy.csv"
    toy_df.to_csv(path, index=False)
    df, schema = load_dataset(str(path), target="Class")
    assert "Class" in df.columns
    assert set(schema.numeric) == {"age", "bmi"}
    assert set(schema.categorical) == {"sex", "smoker"}
    assert schema.task_type == "classification"


def test_split_stratified(toy_df, toy_schema):
    train_df, test_df = split_train_test(toy_df, toy_schema, seed=123, test_size=0.2, stratify=True)
    train_rate = train_df["Class"].mean()
    test_rate = test_df["Class"].mean()
    assert abs(train_rate - test_rate) < 0.1
    assert len(train_df) + len(test_df) == len(toy_df)
