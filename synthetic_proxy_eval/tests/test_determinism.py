"""Determinism tests for repeated pipeline runs."""

from __future__ import annotations

import yaml
import pytest
import pandas as pd

from eval_harness.experiments import run_pipeline


def test_deterministic_outputs(tmp_path, toy_df, test_config):
    pytest.importorskip("pyarrow")
    data_path = tmp_path / "toy.csv"
    cfg_path = tmp_path / "config.yaml"
    outdir_a = tmp_path / "run_a"
    outdir_b = tmp_path / "run_b"

    toy_df.to_csv(data_path, index=False)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(test_config, fh)

    run_pipeline(str(cfg_path), str(data_path), "Class", str(outdir_a))
    run_pipeline(str(cfg_path), str(data_path), "Class", str(outdir_b))

    df_a = pd.read_parquet(outdir_a / "results.parquet")
    df_b = pd.read_parquet(outdir_b / "results.parquet")
    pd.testing.assert_frame_equal(df_a, df_b)
