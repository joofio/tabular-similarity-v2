"""Integration test for CLI pipeline."""

from __future__ import annotations

import yaml
import pytest

from eval_harness import cli


def test_cli_runs_end_to_end(tmp_path, toy_df, test_config):
    pytest.importorskip("pyarrow")
    data_path = tmp_path / "toy.csv"
    config_path = tmp_path / "config.yaml"
    outdir = tmp_path / "run"

    toy_df.to_csv(data_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(test_config, fh)

    cli.main(
        [
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--target",
            "Class",
            "--outdir",
            str(outdir),
        ]
    )

    assert (outdir / "results.parquet").exists()
    assert (outdir / "summary.csv").exists()
    assert (outdir / "utility_long.csv").exists()
    assert (outdir / "proxy_long.csv").exists()
    assert (outdir / "fidelity.csv").exists()
    assert (outdir / "stats.csv").exists()
    assert (outdir / "sensitivity.csv").exists()
    assert (outdir / "invariance.csv").exists()
